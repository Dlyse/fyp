"""
Runs on the Orin. Pushes PX4 telemetry, the planned route and a compressed
depth image (+ obstacle summary) to the relay server over a websocket.

pip install "mavsdk<4" websockets opencv-python numpy pyrealsense2
(pyrealsense2 has no aarch64 wheel on some JetPack versions: build librealsense
with CUDA/RSUSB per Intel's Jetson guide. Use FAKE_DEPTH=1 to test without cameras.)

Env vars:
  SERVER_URL  ws://<server-ip>:8000/ws/drone   (use wss:// once you have TLS)
  DRONE_TOKEN same secret as the server
  MAV_ADDR    udpin://0.0.0.0:14540  (PX4 SITL / mavlink-router)  or  serial:///dev/ttyTHS1:921600
  ROS2=1      take depth/colour from ROS 2 topics (Gazebo) instead of RealSense; edit ROS_TOPICS
"""
import asyncio
import json
import math
import struct
import os
import time

import cv2
import numpy as np
import threading
from collections import deque
import websockets
from mavsdk import System

SERVER_URL = os.environ.get("SERVER_URL", "ws://192.168.1.50:8000/ws/drone")
TOKEN = os.environ.get("DRONE_TOKEN", "change-me")
MAV_ADDR = os.environ.get("MAV_ADDR", "udpin://0.0.0.0:14540")

CELL = os.environ.get("LINK_PROFILE", "wifi") == "cellular"   # LINK_PROFILE=cellular: low-data defaults


def _hz(name, wifi, cell=None):
    return float(os.environ.get(name, cell if (CELL and cell is not None) else wifi))


# Update rates, all adjustable from the command line, e.g.  DEPTH3D_ALL_HZ=2 IMG_HZ=2 python ...
# Defaults suit WiFi / simulation. For a cellular link use something like 1-2 for the image rates.
TELEMETRY_HZ = _hz("TELEMETRY_HZ", 5, 2)
OBS_HZ = _hz("OBS_HZ", 10, 2)      # obstacle summary rate, per camera
IMG_HZ = _hz("IMG_HZ", 5, 1)       # greyscale depth preview rate, per camera
DEPTH_MAX_MM = 6000        # colour-map range
IMG_SIZE = (320, 180)      # width, height sent to the website
JPEG_QUALITY = 60

state: dict = {}           # latest telemetry, filled by the watcher tasks
route: list = []           # [[lat, lon, alt], ...] downloaded from a PX4 mission
route_override: list = []  # planned route published by your flight script on /dashboard/route
mission_msgs: dict = {}    # latest /dashboard/mission message per type: {type: (json_text, time_received)}
obstacle_info: dict = {}   # other code (your avoidance script) can write into this


# ---------------------------------------------------------------- depth input
CAMERAS = [  # (name, role, serial). Leave serial "" to auto-assign in plug-in order;
    ("front", "avoid", ""),   # run once, read the printed serials, then fill them in
    ("right", "avoid", ""),   # so each name always maps to the same physical camera.
    ("back", "avoid", ""),
    ("left", "avoid", ""),
    ("down", "land", ""),
]
RS_W, RS_H, RS_FPS = 424, 240, 15   # depth only: keeps 5 cameras inside USB bandwidth
FAKE_DEPTH = os.environ.get("FAKE_DEPTH") == "1"
USE_COLOR = os.environ.get("USE_COLOR", "1") == "1"   # RGB for the 3D view; 0 = depth only

latest: dict = {}   # camera name -> newest uint16 depth image in mm
INTR: dict = {}     # camera name -> (fx, fy, cx, cy) at native resolution
view3d = None       # camera the website wants as a 3D point cloud (None = off)
rgb_cam = None      # camera the website wants as live RGB video: a name, "all", or None = off
RGB_FPS = _hz("RGB_FPS", 30, 2)                    # frame-rate cap per camera for the RGB video
RGB_QUALITY = int(_hz("RGB_QUALITY", 70, 50))       # JPEG quality of the RGB video feed
latest_rgb: dict = {}                           # camera name -> newest BGR frame
rgb_seq: dict = {}                              # camera name -> frame counter
LOOP = None                                     # asyncio loop and event, set in main()
rgb_event = None


def wantsrgb(name):
    return rgb_cam == name or rgb_cam == "all"


def push_rgb(name, bgr):
    """Called from camera threads: keep only the newest RGB frame and wake the sender."""
    latest_rgb[name] = bgr
    rgb_seq[name] = rgb_seq.get(name, 0) + 1
    if LOOP is not None and rgb_event is not None:
        LOOP.call_soon_threadsafe(rgb_event.set)


DEPTH3D_HZ = _hz("DEPTH3D_HZ", 8, 1)          # one camera chosen in 3D view
DEPTH3D_ALL_HZ = _hz("DEPTH3D_ALL_HZ", 5, 0.5)  # per camera when the page asks for "all"
latest_color: dict = {}  # camera name -> (timestamp, BGR image aligned to the depth image)
ACTIVE: list = []   # (name, role) of cameras actually started; only these are sent


# ------------------------------------------------------------ rolling recorder (Save button)
REC_SECONDS = _hz("REC_SECONDS", 300)            # history kept for the Save button
REC_FPS = _hz("REC_FPS", 2)                      # recording rate per camera (low on purpose)
REC_QUALITY = int(_hz("REC_QUALITY", 60))
REC_DIR = os.path.expanduser(os.environ.get("REC_DIR", "~/recordings"))
rec_buf: dict = {}      # camera -> deque of (time, rgb jpeg or None, depth jpeg or None)
rec_need: dict = {}     # camera -> True while the recorder wants the next colour frame
rec_rgb: dict = {}      # camera -> colour frame captured for the recorder
saving = False
bg_tasks: set = set()
viewers_n = 1           # open dashboard pages, told by the server (0 = send no previews)


def _rec_encode(rgb, depth_mm):
    q = [cv2.IMWRITE_JPEG_QUALITY, REC_QUALITY]
    r = cv2.imencode(".jpg", rgb, q)[1].tobytes() if rgb is not None else None
    d = None
    if depth_mm is not None:                     # same greyscale look as the dashboard
        g = (255 - np.clip(depth_mm.astype(np.float32) / DEPTH_MAX_MM * 225, 0, 225)).astype(np.uint8)
        g[depth_mm == 0] = 0
        d = cv2.imencode(".jpg", g, q)[1].tobytes()
    return r, d


async def recorder():
    """Runs all the time, even with the link down: keeps the last REC_SECONDS of RGB + depth."""
    while True:
        t0 = time.time()
        for name, _ in list(ACTIVE):
            rgb = rec_rgb.pop(name, None)
            rec_need[name] = True                # ask the camera for its next colour frame
            r, d = await asyncio.to_thread(_rec_encode, rgb, latest.get(name))
            buf = rec_buf.setdefault(name, deque())
            if r or d:                           # skip ticks where the camera gave nothing
                buf.append((t0, r, d))
            while buf and t0 - buf[0][0] > REC_SECONDS:
                buf.popleft()
        span = max((b[-1][0] - b[0][0] for b in rec_buf.values() if b), default=0)
        state["rec"] = {"buffered_s": round(span), "saving": saving, "last": state.get("rec", {}).get("last")}
        await asyncio.sleep(max(0, 1 / REC_FPS - (time.time() - t0)))


def save_recording():
    """Write the buffered window to REC_DIR/<date_time>/ as one mp4 per camera and stream."""
    snap = {n: list(b) for n, b in rec_buf.items() if b}
    if not snap:
        raise RuntimeError("no camera frames recorded yet (are the camera topics publishing?)")
    out = os.path.join(REC_DIR, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    total = 0
    for name, frames in snap.items():
        for kind, i in (("rgb", 1), ("depth", 2)):
            jpgs = [f[i] for f in frames if f[i] is not None]
            if not jpgs:
                continue
            imgs = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR) for j in jpgs]
            h, w = imgs[0].shape[:2]
            vw = cv2.VideoWriter(os.path.join(out, f"{name}_{kind}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), REC_FPS, (w, h))
            if vw.isOpened():
                for im in imgs:
                    vw.write(im)
                vw.release()
            else:                                # this OpenCV has no video codec: save JPEG images instead
                folder = os.path.join(out, f"{name}_{kind}")
                os.makedirs(folder, exist_ok=True)
                for k, j in enumerate(jpgs):
                    with open(os.path.join(folder, f"{k:05d}.jpg"), "wb") as fh:
                        fh.write(j)
            total += len(jpgs)
    seconds = max(f[-1][0] - f[0][0] for f in snap.values())
    with open(os.path.join(out, "info.json"), "w") as fh:
        json.dump({"fps": REC_FPS, "seconds": round(seconds), "cameras": list(snap),
                   "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, fh)
    return {"dir": out, "seconds": round(seconds), "frames": total}


async def do_save(ws):
    global saving
    if saving:
        return

    async def say(**kw):
        try:
            await ws.send(json.dumps({"type": "rec_status", **kw}))
        except Exception:
            pass

    saving = True
    print("Save requested from the dashboard...")
    await say(state="saving")
    try:
        info = await asyncio.to_thread(save_recording)
        state.setdefault("rec", {})["last"] = info["dir"]
        print(f"recording saved: {info}")
        await say(state="saved", **info)
    except Exception as e:
        print(f"recording save failed: {e!r}")
        await say(state="error", msg=str(e))
    finally:
        saving = False


def wants3d(name):
    """True if the website currently wants this camera in 3D ("all" = every avoidance camera)."""
    if view3d == name:
        return True
    return view3d == "all" and any(n == name and r == "avoid" for n, r in ACTIVE)


def realsense_reader(name, serial):
    import pyrealsense2 as rs
    align = rs.align(rs.stream.depth)   # resample colour onto the depth image
    use_color = USE_COLOR
    while True:
        pipe = rs.pipeline()
        try:
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.depth, RS_W, RS_H, rs.format.z16, RS_FPS)
            if use_color:
                cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
            try:
                prof = pipe.start(cfg)
            except Exception as e:
                if not use_color:
                    raise
                print(f"{name}: colour stream unavailable ({e!r}), using depth only")
                use_color = False
                continue
            scale = prof.get_device().first_depth_sensor().get_depth_scale()
            k = prof.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            INTR[name] = (k.fx, k.fy, k.ppx, k.ppy)
            print(f"{name}: started ({serial}), colour={'on' if use_color else 'off'}")
            while True:
                fs = pipe.wait_for_frames(5000)
                if use_color and wantsrgb(name):
                    cf = fs.get_color_frame()
                    if cf:
                        push_rgb(name, np.asanyarray(cf.get_data()).copy())
                if use_color and rec_need.get(name):
                    cf = fs.get_color_frame()
                    if cf:
                        rec_rgb[name] = np.asanyarray(cf.get_data()).copy()
                        rec_need[name] = False
                if use_color and wants3d(name):       # only align cameras being viewed in 3D
                    fs = align.process(fs)
                    c = fs.get_color_frame()
                    if c:
                        latest_color[name] = (time.time(), np.asanyarray(c.get_data()).copy())
                raw = np.asanyarray(fs.get_depth_frame().get_data())
                if abs(scale - 0.001) > 1e-6:
                    raw = (raw * (scale * 1000)).astype(np.uint16)
                latest[name] = raw.copy()
        except Exception as e:
            print(f"{name}: camera error {e!r}, restarting")
            latest.pop(name, None)
            latest_color.pop(name, None)
            try:
                pipe.stop()
            except Exception:
                pass
            time.sleep(2)


def fake_reader(name, i):
    h, w = RS_H, RS_W
    f = (w / 2) / math.tan(math.radians(43.5))
    INTR[name] = (f, f, w / 2, h / 2)
    y, x = np.mgrid[0:h, 0:w]
    while True:
        d = 2000 + 1500 * np.sin(x / 40.0 + i) + 800 * np.cos(y / 30.0 + time.time())
        latest[name] = d.astype(np.uint16)
        if wants3d(name) or wantsrgb(name) or rec_need.get(name):   # fake colour image for testing
            c = np.zeros((h, w, 3), np.uint8)
            c[:, :, 0] = (x * 255 // w).astype(np.uint8)
            c[:, :, 1] = (y * 255 // h).astype(np.uint8)
            c[:, :, 2] = 160
            if wants3d(name):
                latest_color[name] = (time.time(), c)
            if wantsrgb(name):
                push_rgb(name, c)
            if rec_need.get(name):
                rec_rgb[name] = c
                rec_need[name] = False
        time.sleep(1 / RS_FPS)


# ------------------------------------------------------- ROS 2 / Gazebo camera input
# Edit to match `ros2 topic list`:  name: (depth image, camera info, colour image or None)
# Depth may be 16UC1 (mm) or 32FC1 (m). Colour should be the same size/pose as depth (as an
# RGBD sensor gives) so the colours line up with the points.
ROS_TOPICS = {
    "front": ("/cam_front/depth_image", "/cam_front/camera_info", "/cam_front/image"),
    "right": ("/cam_right/depth_image", "/cam_right/camera_info", "/cam_right/image"),
    "back": ("/cam_back/depth_image", "/cam_back/camera_info", "/cam_back/image"),
    "left": ("/cam_left/depth_image", "/cam_left/camera_info", "/cam_left/image"),
    # the sim has no downward (landing) camera, so there is no "down" entry
}


def _img(msg, dtype, ch=1):
    """sensor_msgs/Image -> numpy array (no cv_bridge needed), honouring row padding."""
    a = np.frombuffer(msg.data, dtype)
    per_row = msg.step // np.dtype(dtype).itemsize
    a = a.reshape(msg.height, per_row)[:, : msg.width * ch]
    return a.reshape(msg.height, msg.width, ch) if ch > 1 else a


def ros_depth_to_mm(msg):
    if msg.encoding in ("16UC1", "mono16"):
        return _img(msg, np.uint16).copy()
    if msg.encoding == "32FC1":
        a = np.nan_to_num(_img(msg, np.float32), nan=0.0, posinf=0.0, neginf=0.0) * 1000
        return np.clip(a, 0, 65535).astype(np.uint16)
    return None


def ros_color_to_bgr(msg):
    if msg.encoding in ("rgb8", "bgr8"):
        a = _img(msg, np.uint8, 3)
        return (a[:, :, ::-1] if msg.encoding == "rgb8" else a).copy()
    if msg.encoding in ("rgba8", "bgra8"):
        a = _img(msg, np.uint8, 4)[:, :, :3]
        return (a[:, :, ::-1] if msg.encoding == "rgba8" else a).copy()
    return None


def route_from_json(text):
    """Planned route from the flight script, as JSON on the ROS 2 topic /dashboard/route:
    {"frame": "latlon", "points": [[lat, lon, alt_m], ...]}   or
    {"frame": "ned",    "points": [[north_m, east_m, down_m], ...]}  (local PX4 frame, like trajectory_setpoint)
    Returns [[lat, lon, alt_m], ...], or None if local points can't be placed yet (no position)."""
    d = json.loads(text)
    pts = d["points"]
    if d.get("frame", "latlon") == "ned":
        pos, ned = state.get("pos"), state.get("ned")
        if not pos or not ned:
            return None
        lat0 = pos["lat"] - ned["n"] / 110540            # lat/lon of the local origin
        lon0 = pos["lon"] - ned["e"] / (111320 * math.cos(math.radians(pos["lat"])))
        k = 111320 * math.cos(math.radians(lat0))
        return [[lat0 + p[0] / 110540, lon0 + p[1] / k, -p[2] if len(p) > 2 else 0.0] for p in pts]
    return [[float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0] for p in pts]


def start_ros2():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    rclpy.init()
    node = Node("drone_link")
    roles = {c[0]: c[1] for c in CAMERAS}
    for name, (depth_t, info_t, color_t) in ROS_TOPICS.items():
        ACTIVE.append((name, roles.get(name, "avoid")))

        def on_depth(msg, name=name):
            d = ros_depth_to_mm(msg)
            if d is None:
                print(f"{name}: unsupported depth encoding {msg.encoding}")
            else:
                latest[name] = d

        def on_info(msg, name=name):
            INTR[name] = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

        def on_color(msg, name=name):
            want3, wantrgb, needrec = wants3d(name), wantsrgb(name), rec_need.get(name, False)
            if not (want3 or wantrgb or needrec):
                return                      # nobody needs it: skip the conversion
            c = ros_color_to_bgr(msg)
            if c is None:
                return
            if want3:
                latest_color[name] = (time.time(), c)
            if wantrgb:
                push_rgb(name, c)
            if needrec:
                rec_rgb[name] = c
                rec_need[name] = False

        node.create_subscription(Image, depth_t, on_depth, qos_profile_sensor_data)
        node.create_subscription(CameraInfo, info_t, on_info, qos_profile_sensor_data)
        if color_t:
            node.create_subscription(Image, color_t, on_color, qos_profile_sensor_data)
        print(f"{name}: ROS 2 topics {depth_t}, {info_t}, {color_t}")
    from std_msgs.msg import String

    def on_route(msg):
        global route_override
        try:
            r = route_from_json(msg.data)
            if r is not None:
                if r != route_override:
                    print(f"route from /dashboard/route: {len(r)} waypoints")
                route_override = r
        except Exception as e:
            print("bad /dashboard/route message:", e)

    node.create_subscription(String, "/dashboard/route", on_route, 10)
 
    def on_mission(msg):                                   
        try:                                               
            mission_msgs[json.loads(msg.data)["type"]] = (msg.data, time.time())  
        except Exception as e:                             
            print("bad /dashboard/mission message:", e)    

    node.create_subscription(String, "/dashboard/mission", on_mission, 10)    

    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()


def start_cameras():
    if os.environ.get("ROS2") == "1":
        start_ros2()
        return
    if FAKE_DEPTH:
        for i, (name, role, _) in enumerate(CAMERAS):
            ACTIVE.append((name, role))
            threading.Thread(target=fake_reader, args=(name, i), daemon=True).start()
        return
    import pyrealsense2 as rs
    found = [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]
    print("RealSense serials found:", found)
    spare = [s for s in found if s not in {c[2] for c in CAMERAS if c[2]}]
    for name, role, serial in CAMERAS:
        serial = serial or (spare.pop(0) if spare else None)
        if not serial:
            print(f"{name}: no camera available")
            continue
        print(f"{name} -> {serial}")
        ACTIVE.append((name, role))
        threading.Thread(target=realsense_reader, args=(name, serial), daemon=True).start()
        time.sleep(1)   # stagger USB start-up


def encode_depth3d(name, depth_mm):
    """Half-res lossless depth PNG (R = high byte, G = low byte) + optional colour JPEG.
    Payload = 4-byte PNG length + PNG + JPEG (JPEG empty if no colour)."""
    h, w = depth_mm.shape
    tw, th = w // 2, h // 2
    small = cv2.resize(depth_mm, (tw, th), interpolation=cv2.INTER_NEAREST)
    img = np.zeros((th, tw, 3), np.uint8)
    img[:, :, 2] = small >> 8
    img[:, :, 1] = small & 255
    ok, png = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    jpg = b""
    c = latest_color.get(name)
    if c and time.time() - c[0] < 1.0:
        col = cv2.resize(c[1], (tw, th), interpolation=cv2.INTER_AREA)
        ok2, j = cv2.imencode(".jpg", col, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok2:
            jpg = j.tobytes()
    f = (w / 2) / math.tan(math.radians(43.5))
    fx, fy, cx, cy = INTR.get(name, (f, f, w / 2, h / 2))
    s = tw / w
    meta = {"type": "depth3d", "cam": name, "w": tw, "h": th, "color": bool(jpg),
            "fx": fx * s, "fy": fy * s, "cx": cx * s, "cy": cy * s}
    if not ok:
        return meta, None
    png = png.tobytes()
    return meta, struct.pack(">I", len(png)) + png + jpg


def _near(a: np.ndarray):
    v = a[a > 0]
    return round(float(np.percentile(v, 2)) / 1000, 2) if v.size else None


def process_depth(name, role, depth_mm, want_image):
    h, w = depth_mm.shape
    summary = {"type": "obstacle", "cam": name, "role": role, "t": time.time(),
               "min_m": _near(depth_mm)}
    if role == "land":
        c = depth_mm[h // 3:2 * h // 3, w // 3:2 * w // 3]
        v = c[c > 0]
        summary["ground_m"] = round(float(np.median(v)) / 1000, 2) if v.size else None
    else:
        t3 = [depth_mm[:, i * w // 3:(i + 1) * w // 3] for i in range(3)]
        summary.update(left_m=_near(t3[0]), center_m=_near(t3[1]), right_m=_near(t3[2]))
    if not want_image:
        return summary, None
    small = cv2.resize(depth_mm, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    # greyscale: near = bright, far = dark, no reading = black
    vis = (255 - np.clip(small.astype(np.float32) / DEPTH_MAX_MM * 225, 0, 225)).astype(np.uint8)
    vis[small == 0] = 0
    ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return summary, (jpg.tobytes() if ok else None)


# ------------------------------------------------------------ PX4 telemetry
async def watch(stream, key, fn):
    async for x in stream:
        state[key] = fn(x)


async def start_px4():
    drone = System()
    await drone.connect(system_address=MAV_ADDR)
    async for s in drone.core.connection_state():
        if s.is_connected:
            break
    print("PX4 connected")
    t = drone.telemetry
    for coro in (
        watch(t.position(), "pos", lambda p: {"lat": p.latitude_deg, "lon": p.longitude_deg,
                                              "alt_m": p.relative_altitude_m}),
        watch(t.position_velocity_ned(), "ned",
              lambda v: {"n": v.position.north_m, "e": v.position.east_m}),
        watch(t.attitude_euler(), "att", lambda a: {"roll": a.roll_deg, "pitch": a.pitch_deg,
                                                     "yaw": a.yaw_deg}),
        watch(t.velocity_ned(), "vel", lambda v: {"n": v.north_m_s, "e": v.east_m_s, "d": v.down_m_s}),
        watch(t.battery(), "batt", lambda b: {"v": b.voltage_v, "pct": b.remaining_percent}),
        watch(t.flight_mode(), "mode", lambda m: str(m)),
        watch(t.armed(), "armed", bool),
        watch(t.in_air(), "in_air", bool),
        watch(t.gps_info(), "gps", lambda g: {"sats": g.num_satellites, "fix": str(g.fix_type)}),
        watch(drone.mission.mission_progress(), "mission",
              lambda p: {"current": p.current, "total": p.total}),
    ):
        asyncio.create_task(coro)
    return drone


async def refresh_route(drone):
    global route
    while True:
        try:
            plan = await drone.mission.download_mission()
            route = [[i.latitude_deg, i.longitude_deg, i.relative_altitude_m]
                     for i in plan.mission_items]
        except Exception:
            pass   # no mission on the vehicle yet
        await asyncio.sleep(10)


# ----------------------------------------------------------------- senders
async def send_state(ws):
    n, last_route = 0, None
    sent = {}                                                                  
    while True:
        await ws.send(json.dumps({"type": "telemetry", "t": time.time(), "avoidance": obstacle_info, **state}))
        current = route_override or route
        if current != last_route or n % int(TELEMETRY_HZ * 5) == 0:   # as soon as it changes, then every ~5 s
            await ws.send(json.dumps({"type": "route", "points": current}))
            last_route = list(current)
        for typ, (text, got) in list(mission_msgs.items()):                   
            prev, t_prev = sent.get(typ, (None, 0.0))                          
            fresh = typ != "mission" or time.time() - got < 3.0                #nothing from a dead flight script
            if fresh and (text != prev or (typ == "mission" and time.time() - t_prev > 2.0)):   
                await ws.send(text)                                            
                sent[typ] = (text, time.time())                               
        n += 1
        await asyncio.sleep(1 / TELEMETRY_HZ)


async def send_depth(ws):
    await ws.send(json.dumps({"type": "cameras", "names": [c[0] for c in ACTIVE]}))
    last_obs, last_img, last3d = {}, {}, {}
    while True:
        t0 = time.time()
        if viewers_n == 0:                 # nobody has the dashboard open: save the data
            await asyncio.sleep(0.25)
            continue
        names = [c[0] for c in ACTIVE]
        jobs = []
        for idx, (name, role) in enumerate(ACTIVE):
            frame = latest.get(name)
            if frame is None:
                continue
            want_obs = t0 - last_obs.get(name, 0) >= 1 / OBS_HZ
            want_img = t0 - last_img.get(name, 0) >= 1 / IMG_HZ
            if want_obs or want_img:       # each camera has its own timers
                if want_obs:
                    last_obs[name] = t0
                if want_img:
                    last_img[name] = t0
                jobs.append((idx, name, role, frame, want_img))
        if jobs:
            results = await asyncio.to_thread(
                lambda: [(idx, *process_depth(n, r, f, w)) for idx, n, r, f, w in jobs])
            for idx, summary, jpg in results:
                await ws.send(json.dumps(summary))
                if jpg:
                    await ws.send(bytes([idx, 0]) + jpg)   # header: camera index, kind 0 = preview JPEG
        rate = DEPTH3D_ALL_HZ if view3d == "all" else DEPTH3D_HZ
        for n in [n for n in names if wants3d(n) and n in latest]:
            if t0 - last3d.get(n, 0) < 1 / rate:
                continue
            last3d[n] = t0
            meta, payload = await asyncio.to_thread(encode_depth3d, n, latest[n])
            if payload:
                await ws.send(json.dumps(meta))
                await ws.send(bytes([names.index(n), 1]) + payload)   # kind 1 = depth + colour
        await asyncio.sleep(max(0, 1 / 60 - (time.time() - t0)))      # loop ticks at 60 Hz


async def send_rgb(ws):
    """Live RGB video: frame-driven, newest frame per camera only, JPEG at native resolution."""
    last_t, sent = {}, {}
    while True:
        await rgb_event.wait()
        rgb_event.clear()
        now = time.time()
        due = []
        for idx, (name, _) in enumerate(ACTIVE):
            frame, seq = latest_rgb.get(name), rgb_seq.get(name, 0)
            if not wantsrgb(name) or frame is None or seq == sent.get(name):
                continue
            if now - last_t.get(name, 0) < 0.9 / RGB_FPS:      # per-camera frame-rate cap
                continue
            last_t[name], sent[name] = now, seq
            due.append((idx, frame))
        if not due:
            continue
        results = await asyncio.gather(*[
            asyncio.to_thread(cv2.imencode, ".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, RGB_QUALITY])
            for _, f in due])
        for (idx, _), (ok, jpg) in zip(due, results):
            if ok:
                await ws.send(bytes([idx, 2]) + jpg.tobytes())   # kind 2 = RGB video


async def recv_loop(ws):
    global view3d, rgb_cam, viewers_n
    async for msg in ws:
        if isinstance(msg, str):
            try:
                m = json.loads(msg)
            except ValueError:
                continue
            if m.get("type") == "view3d":
                view3d = m.get("cam")
                # print("3D view:", view3d)
            elif m.get("type") == "viewers":
                viewers_n = int(m.get("n", 1))
            elif m.get("type") == "rec_save":
                task = asyncio.create_task(do_save(ws))
                bg_tasks.add(task)
                task.add_done_callback(bg_tasks.discard)
            elif m.get("type") == "rgb":
                rgb_cam = m.get("cam")
                # print("RGB view:", rgb_cam)


async def run_link():
    backoff = 1
    while True:
        try:
            async with websockets.connect(f"{SERVER_URL}?token={TOKEN}", max_size=None,
                                          ping_interval=10, ping_timeout=10) as ws:
                print("link up")
                backoff = 1
                tasks = [asyncio.create_task(send_state(ws)), asyncio.create_task(send_depth(ws)),
                         asyncio.create_task(recv_loop(ws)), asyncio.create_task(send_rgb(ws))]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
                for p in pending:
                    p.cancel()
                for d in done:
                    d.result()   # re-raise
        except Exception as e:
            print(f"link down ({e!r}), retrying in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15)


async def fake_px4():
    """NO_PX4=1: fake telemetry and route so the link can be tested without a flight controller."""
    global route
    lat0, lon0 = 1.3000, 103.8000
    route = [[lat0 + dy, lon0 + dx, 20.0]
             for dx, dy in ((0, 0), (0.0006, 0), (0.0006, 0.0006), (0, 0.0006))]
    t0 = time.time()
    while True:
        a = (time.time() - t0) / 20 * 2 * math.pi
        state.update(
            pos={"lat": lat0 + 0.0003 + 0.0003 * math.sin(a),
                 "lon": lon0 + 0.0003 + 0.0003 * math.cos(a), "alt_m": 20.0},
            att={"roll": 0.0, "pitch": 0.0, "yaw": math.degrees(a) % 360},
            vel={"n": 0.0, "e": 0.0, "d": 0.0}, batt={"v": 16.0, "pct": 0.8},
            mode="FAKE", armed=True, in_air=True, gps={"sats": 12, "fix": "FAKE"},
            mission={"current": 1, "total": 4})
        await asyncio.sleep(0.5)


async def px4_task():
    """Connect to PX4 in the background so camera data flows even with no flight controller."""
    try:
        drone = await start_px4()
        await refresh_route(drone)
    except Exception as e:
        print(f"PX4 task stopped: {e!r}")


async def main():
    global LOOP, rgb_event
    LOOP = asyncio.get_running_loop()
    rgb_event = asyncio.Event()
    await asyncio.to_thread(start_cameras)
    asyncio.create_task(recorder())
    if os.environ.get("NO_PX4") == "1":
        asyncio.create_task(fake_px4())
    else:
        asyncio.create_task(px4_task())
    await run_link()


if __name__ == "__main__":
    asyncio.run(main())

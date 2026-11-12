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
"""
import asyncio
import json
import math
import os
import time

import cv2
import numpy as np
import threading
import websockets
from mavsdk import System

SERVER_URL = os.environ.get("SERVER_URL", "ws://192.168.1.50:8000/ws/drone")
TOKEN = os.environ.get("DRONE_TOKEN", "change-me")
MAV_ADDR = os.environ.get("MAV_ADDR", "udpin://0.0.0.0:14540")

TELEMETRY_HZ = 2
OBS_HZ = 5                # obstacle summary rate, per camera
IMG_HZ = 2                # depth image rate, per camera (keeps cellular use low)
DEPTH_MAX_MM = 6000        # colour-map range
IMG_SIZE = (320, 180)      # width, height sent to the website
JPEG_QUALITY = 60

state: dict = {}           # latest telemetry, filled by the watcher tasks
route: list = []           # [[lat, lon, alt], ...]
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

latest: dict = {}   # camera name -> newest uint16 depth image in mm
ACTIVE: list = []   # (name, role) of cameras actually started; only these are sent


def realsense_reader(name, serial):
    import pyrealsense2 as rs
    while True:
        pipe = rs.pipeline()
        try:
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.depth, RS_W, RS_H, rs.format.z16, RS_FPS)
            prof = pipe.start(cfg)
            scale = prof.get_device().first_depth_sensor().get_depth_scale()
            print(f"{name}: started ({serial})")
            while True:
                fs = pipe.wait_for_frames(5000)
                raw = np.asanyarray(fs.get_depth_frame().get_data())
                if abs(scale - 0.001) > 1e-6:
                    raw = (raw * (scale * 1000)).astype(np.uint16)
                latest[name] = raw.copy()
        except Exception as e:
            print(f"{name}: camera error {e!r}, restarting")
            latest.pop(name, None)
            try:
                pipe.stop()
            except Exception:
                pass
            time.sleep(2)


def fake_reader(name, i):
    h, w = RS_H, RS_W
    y, x = np.mgrid[0:h, 0:w]
    while True:
        d = 2000 + 1500 * np.sin(x / 40.0 + i) + 800 * np.cos(y / 30.0 + time.time())
        latest[name] = d.astype(np.uint16)
        time.sleep(1 / RS_FPS)


def start_cameras():
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
    vis = np.clip(small.astype(np.float32) / DEPTH_MAX_MM * 255, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
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
    n = 0
    while True:
        await ws.send(json.dumps({"type": "telemetry", "t": time.time(), "avoidance": obstacle_info, **state}))
        if n % (TELEMETRY_HZ * 5) == 0:
            await ws.send(json.dumps({"type": "route", "points": route}))
        n += 1
        await asyncio.sleep(1 / TELEMETRY_HZ)


async def send_depth(ws):
    await ws.send(json.dumps({"type": "cameras", "names": [c[0] for c in ACTIVE]}))
    last_img = {}
    while True:
        t0 = time.time()
        jobs = []
        for idx, (name, role) in enumerate(ACTIVE):
            frame = latest.get(name)
            if frame is None:
                continue
            want = t0 - last_img.get(name, 0) >= 1 / IMG_HZ
            if want:
                last_img[name] = t0
            jobs.append((idx, name, role, frame, want))
        results = await asyncio.to_thread(
            lambda: [(idx, *process_depth(n, r, f, w)) for idx, n, r, f, w in jobs])
        for idx, summary, jpg in results:
            await ws.send(json.dumps(summary))
            if jpg:
                await ws.send(bytes([idx]) + jpg)   # first byte = camera index
        await asyncio.sleep(max(0, 1 / OBS_HZ - (time.time() - t0)))


async def run_link():
    backoff = 1
    while True:
        try:
            async with websockets.connect(f"{SERVER_URL}?token={TOKEN}", max_size=None,
                                          ping_interval=10, ping_timeout=10) as ws:
                print("link up")
                backoff = 1
                tasks = [asyncio.create_task(send_state(ws)), asyncio.create_task(send_depth(ws))]
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
    await asyncio.to_thread(start_cameras)
    if os.environ.get("NO_PX4") == "1":
        asyncio.create_task(fake_px4())
    else:
        asyncio.create_task(px4_task())
    await run_link()


if __name__ == "__main__":
    asyncio.run(main())

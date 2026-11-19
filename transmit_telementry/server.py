"""
Relay server: the drone connects OUT to /ws/drone, browsers connect to /ws/view.
Local test:  DRONE_TOKEN=secret uvicorn server:app --host 0.0.0.0 --port 8000
Cloud:       also set VIEWER_PASSWORD (login for the dashboard) and run behind HTTPS (see Caddyfile).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from urllib.parse import parse_qs

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

TOKEN = os.environ.get("DRONE_TOKEN", "change-me")                  # the drone's secret
VIEWER_PASSWORD = os.environ.get("VIEWER_PASSWORD", "")             # empty = no login (local testing only)
HERE = os.path.dirname(os.path.abspath(__file__))
if VIEWER_PASSWORD and TOKEN == "change-me":
    print("WARNING: set DRONE_TOKEN to a long random string")

app = FastAPI()
viewers: set[WebSocket] = set()
last_json: dict[str, str] = {}      # latest message of each type, replayed to new viewers
drone_conn: WebSocket | None = None


def _off(kind):
    return json.dumps({"type": kind, "cam": None})


wants: dict[str, str] = {"view3d": _off("view3d"), "rgb": _off("rgb")}   # latest choice per kind
last_frames: dict[tuple[int, int], bytes] = {}  # latest JPEG per (camera index, kind)

# ------------------------------------------------------------------ login (stateless cookie)
LOGIN = ('<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1">'
         '<body style="font-family:system-ui;background:#111;color:#eee;display:grid;place-items:center;height:100vh">'
         '<form method="post" action="/login"><input type="password" name="password" placeholder="Password" '
         'autofocus style="padding:8px"> <button style="padding:8px">Log in</button><!--msg--></form>')


def _cookie_value():
    return hmac.new(TOKEN.encode(), b"viewer:" + VIEWER_PASSWORD.encode(), hashlib.sha256).hexdigest()


def _authed(cookies):
    return not VIEWER_PASSWORD or hmac.compare_digest(cookies.get("viewer", "").encode(), _cookie_value().encode())


@app.get("/")
async def index(request: Request):
    if not _authed(request.cookies):
        return HTMLResponse(LOGIN)
    return FileResponse(os.path.join(HERE, "index.html"), headers={"Cache-Control": "no-store"})


@app.post("/login")
async def login(request: Request):
    form = parse_qs((await request.body()).decode(errors="ignore"))
    given = form.get("password", [""])[0]
    if VIEWER_PASSWORD and hmac.compare_digest(given.encode(), VIEWER_PASSWORD.encode()):
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie("viewer", _cookie_value(), httponly=True, samesite="strict", max_age=30 * 86400,
                        secure=request.headers.get("x-forwarded-proto") == "https")
        return resp
    return HTMLResponse(LOGIN.replace("<!--msg-->", "<p>Wrong password</p>"), status_code=401)


# ------------------------------------------------------------------ relay
async def broadcast(msg):
    dead = []
    for v in list(viewers):
        try:
            if isinstance(msg, bytes):
                await v.send_bytes(msg)
            else:
                await v.send_text(msg)
        except Exception:
            dead.append(v)
    for v in dead:
        viewers.discard(v)


async def send_drone(text):
    if drone_conn:
        try:
            await drone_conn.send_text(text)
        except Exception:
            pass


async def notify_viewers():       # the drone only sends previews/video while somebody is watching
    await send_drone(json.dumps({"type": "viewers", "n": len(viewers)}))


@app.websocket("/ws/drone")
async def drone_ws(ws: WebSocket, token: str = ""):
    global drone_conn
    if not hmac.compare_digest(token.encode(), TOKEN.encode()):
        await ws.close(code=1008)
        return
    await ws.accept()
    last_json.clear()
    last_frames.clear()
    print("drone connected")
    drone_conn = ws
    for text in wants.values():     # tell the drone what the page is currently showing
        await ws.send_text(text)
    await notify_viewers()
    try:
        while True:
            m = await ws.receive()
            if m["type"] == "websocket.disconnect":
                break
            if m.get("bytes") is not None:
                b = m["bytes"]
                last_frames[(b[0], b[1])] = b
                await broadcast(b)
            elif m.get("text") is not None:
                text = m["text"]
                try:
                    d = json.loads(text)
                    last_json[d["type"] + ":" + d.get("cam", "")] = text
                except (ValueError, KeyError):
                    continue
                await broadcast(text)
    except WebSocketDisconnect:
        pass
    finally:
        print("drone disconnected")
        if drone_conn is ws:
            drone_conn = None
        await broadcast(json.dumps({"type": "link", "up": False}))


@app.websocket("/ws/view")
async def view_ws(ws: WebSocket):
    if not _authed(ws.cookies):
        await ws.close(code=1008)
        return
    await ws.accept()
    for t in last_json.values():
        await ws.send_text(t)
    for b in last_frames.values():
        await ws.send_bytes(b)
    viewers.add(ws)
    await notify_viewers()
    try:
        while True:
            text = await ws.receive_text()
            try:
                msg = json.loads(text)
            except ValueError:
                continue
            kind = msg.get("type")
            if kind in wants:                   # 3D / RGB camera choices: remember and forward
                wants[kind] = text
                await send_drone(text)
            elif kind == "rec_save":            # "Save last 5 min" button: one-shot command
                if drone_conn:
                    await send_drone(text)
                else:
                    await ws.send_text(json.dumps({"type": "rec_status", "state": "error",
                                                   "msg": "drone not connected"}))
    except WebSocketDisconnect:
        pass
    finally:
        viewers.discard(ws)
        await notify_viewers()
        if not viewers:       # last page closed: tell the drone to stop sending 3D / video data
            for kind in wants:
                wants[kind] = _off(kind)
                await send_drone(wants[kind])

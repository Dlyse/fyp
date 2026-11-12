"""
Relay server: drone connects OUT to /ws/drone, browsers connect to /ws/view.
Run:  DRONE_TOKEN=yourSecret uvicorn server:app --host 0.0.0.0 --port 8000
Open: http://<server-ip>:8000/
"""
from __future__ import annotations

import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

TOKEN = os.environ.get("DRONE_TOKEN", "change-me")
HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()
viewers: set[WebSocket] = set()
last_json: dict[str, str] = {}      # latest message of each type, replayed to new viewers
drone_conn: WebSocket | None = None
want3d = json.dumps({"type": "view3d", "cam": None})
last_frames: dict[tuple[int, int], bytes] = {}  # latest JPEG per camera (first byte = camera index)


@app.get("/")
async def index():
    return FileResponse(os.path.join(HERE, "index.html"))


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


@app.websocket("/ws/drone")
async def drone_ws(ws: WebSocket, token: str = ""):
    global drone_conn
    if token != TOKEN:
        await ws.close(code=1008)
        return
    await ws.accept()
    last_json.clear()
    last_frames.clear()
    print("drone connected")
    drone_conn = ws
    await ws.send_text(want3d)
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
    global want3d
    await ws.accept()
    for t in last_json.values():
        await ws.send_text(t)
    for b in last_frames.values():
        await ws.send_bytes(b)
    viewers.add(ws)
    try:
        while True:
            text = await ws.receive_text()
            try:
                msg = json.loads(text)
            except ValueError:
                continue
            if msg.get("type") == "view3d":     # forward the viewer's 3D camera choice to the drone
                want3d = text
                if drone_conn:
                    try:
                        await drone_conn.send_text(text)
                    except Exception:
                        pass
    except WebSocketDisconnect:
        pass
    finally:
        viewers.discard(ws)

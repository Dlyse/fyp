#!/usr/bin/env python3

"""PX4 waypoint follower v7.0 (v6.4 + moving-obstacle robustness fixes, see "v7 changes" below): flies a route and goes around encountered obstacles using its cameras,
including obstacles that MOVE.

Route: spawn point, to a point ~40 m east.

How avoidance works (all in the route frame: forward along the line, sideways = "lanes"):
  * Each camera's clusterer publishes confirmed obstacle boxes in the drone body frame.
    The follower merges the four lists and uses them for every decision below.

  * Forward speed fades towards zero between slow_dist and stop_dist when something is in the
    corridor ahead (half-width body_half_width); it counts as blocked at stop_dist + 0.5 m.
    It must stay clear for clear_time before the drone moves on again, so flickering detections cannot make it creep forward.

  * If it stays blocked for pause_time, it looks for a lane (a sideways offset from the
    line, in steps of lane_step up to max_offset). A lane is only chosen if
      - the sideways move to it is clear (including a pass_margin behind the drone),
      - the camera on that side is running (unless allow_blind_dodge),
      - the corridor ahead in that lane is clear for plan_dist.
    The cheapest acceptable lane wins (small move, close to the line). If left and right are
    about equally good (a real 50/50), left wins (see left_bias).

  * Sideways moves are re-checked every tick and stop if something appears.

  * Once the path ahead is clear it flies on in the new lane. It returns to the line only
    when the sideways move back is clear AND the line is clear ahead, so it never cuts
    back into the obstacle it is passing.

  * Cross-track error is always measured against the ORIGINAL line.

Moving obstacles (new in v6):
  * Every box from the cameras is matched to the same thing seen a moment ago (a "track"). The track's recent
    positions (in the world frame, so the drone's own movement does not count) give its speed. Faster than
    mv_speed_on for a moment = a MOVING obstacle. (The camera clusterer must be able to follow fast things:
    run it with a bigger -p track_gate, see the notes at the bottom of this docstring.)

  * Moving obstacles are kept OUT of the normal obstacle memory. Memory assumes things stay where they were,
    so a moving thing would leave a trail of fake obstacles behind it. Instead we predict where it will be for
    the next mv_horizon seconds (with more spare room the further ahead we look) and test the drone's path
    against that. Things that stand still use the memory exactly as before.

  * Every tick it tries forward speeds of 100%, 75%, 50%, 25% and 0% and keeps the fastest one with no
    predicted crash.
      - 100% is fine: carry on.
      - Only a slower speed is fine: slow down, or stop and let it pass.
      - Even 0% (hovering) would get hit: DODGE NOW. It does not wait for pause_time. It slides to the cheapest
        lane that stays clear of the prediction, at evade_lat_speed. A sideways move that becomes
        unsafe is cancelled and planned again.

  * A moving obstacle that keeps TURNING ROUND (swaying, circling, pacing) cannot be predicted in a straight
    line: the prediction would change every time it turns and the drone would chase it from lane to lane. So once a
    moving thing has been seen to turn round, everywhere it has been in the last mv_sway_window seconds becomes ONE
    fixed no-go box, and the drone picks a lane beyond it. It only counts as standing still again after being
    slow for mv_sway_calm seconds (a sway is slow for a moment at each end).

  * WAIT AND LEARN. A moving obstacle that crosses the route sideways might be a one-off, or it might be
    pacing up and down. The drone cannot tell which until it has seen the thing turn round, and it must never
    start a detour (or a dash past) based on a guess about how far the thing travels. So for the first
    mv_settle_time seconds after it first sees a sideways mover, and for as long as a thing keeps turning round,
    the thing counts as a no-go STRIPE. An end of the stripe is only believed once the thing was seen to turn round
    there; an end we have not seen is assumed to go on for mv_unknown_ext metres. The drone therefore holds back
    and watches, and picks a lane once both ends are known (a lane beyond an end we HAVE seen also works).
    If the ends are never seen (the thing goes out of camera range) it keeps waiting and, after give_up_time,
    recentres and flies home like for any other blocked route.

  * Found in real flight logs: (1) the sideways-move re-plan could flip between lanes 20 times a second, so a
    re-plan now needs mv_replan_gap seconds between two changes, and the test for "is my goal lane still
    bad?" uses the same rules as the test for choosing a lane (only the extra slack differs, so it has some
    hysteresis); (2) with a moving obstacle swaying right in front of it and no lane to dodge into, the
    drone just hovered 1 m from it until it gave up. Now, if even hovering would get hit and no lane is
    safe, it BACKS AWAY along the route at mv_retreat_speed until hovering is safe again.

  * Found in a third real flight log: the same sphere kept getting NEW track numbers about every second
    (several at once), so it never had a history to work out its speed and sway from, and the standing-obstacle
    code (which cannot tell it moves) dodged towards the direction the ball was heading. Now:
      - boxes from all cameras seen at the same moment are merged when they touch (one thing seen from two
        cameras is one thing), mv_fuse_gap;
      - a moving thing that disappears for a moment is matched again when it shows up, even if it has moved
        on by roughly its speed times the time it was gone;
      - a thing that is new (younger than mv_probe_time) gets time to show whether it moves, before the
        drone picks a detour around it as if it were standing still;
      - every 5 s it logs how many boxes each camera gave, so you can see whether the cameras keep dropping it.

  * A no-go zone that is entirely BEHIND the drone no longer blocks it from flying on: on the way home the
    sphere is behind the drone, and the safety margin around the zone used to count as a crash even while
    the drone was flying away from it (logs 1 and 2: it stood still until it gave up and landed in place).

  * It keeps mv_brake_room of extra room in front of itself whenever it moves, so it can stop before the safety margin.
    A detour lane only counts as safe if the drone can slide to it AND then carry on forward along it
    (otherwise "stay where we are" always looked like the best lane in front of a fixed no-go zone).

  * The cameras only see the near side of an object, so every moving obstacle's box is made mv_hidden_back bigger.

  * Set mv_enabled:=false to switch all of this off (then it behaves like v4 + the left bias).

v8 (dashboard): publishes what it is doing as JSON on /dashboard/mission (String): following the route, avoiding an
obstacle, giving up and returning to base, and how the mission ended ("arrived", "returned" = destination unapproachable,
"aborted" = landed in place). When it ends it also publishes the flown path as a "mission_complete" message and saves it to
a JSON file (param flown_route_dir). drone_client.py forwards both to the dashboard.

v7 changes (found by replaying the sphere flights; sparse detections ~1-2 per second, first sighting only 1-5 m away):
  * Speed from SPARSE detections: the old fit needed 4 boxes within 0.8 s, which a 1-2 Hz detector almost never gives,
    so a swaying sphere stayed "standing" for ~14 s (flight 3) and got new track numbers every second. Now the speed
    window is mv_window (1.8 s), 3 boxes are enough, and 2 boxes >= 0.6 m apart also give a speed. Matching a box to a
    track allows mv_search_speed * time-since-seen of movement, so a fast thing is still the same track after a gap.
  * A track that moves ACROSS the route is remembered as a sway stripe for mv_sway_memory seconds, even after
    mv_settle_time, and a new box that appears inside that stripe re-joins the old track (flight 3: after 12 s the stripe
    was dropped, "Path ahead is clear" and the drone flew into the sphere at +4.5 m).
  * STANDOFF: when something (standing, moving or a sway stripe) is closer than standoff / mv_standoff ahead, the drone
    BACKS AWAY along the route (mv_retreat_speed) instead of hovering next to it, as long as the camera behind is
    running and the way back is clear. The brake carrot no longer sits 0.5 m IN FRONT of the drone when something is
    close (flight 3 hovered with the obstacle 0.3 m ahead and the carrot 0.5 m ahead of it).
  * Ends of a sway stripe get mv_end_slack extra metres (sparse boxes miss the true turning point).

Recovery (when blocked with no way around for give_up_time):
  * RECENTER: slide sideways back to the route line at lat_speed.

  * RTH: turn around, hold for 2s so the cameras refill in the new frame, then start flying home.
    The return flight uses the same lane-dodging logic as the forward flight, so obstacles
    on the way back are avoided the same. The waypoint list is built by reversing the original
    route from the current leg backward, ending at home_pos.

  * If RTH itself gets blocked with no way around for give_up_time, the drone lands in place.

Notes:
  * RTH assumes home is reachable via the reversed waypoints; for a single-leg route like
    the default (spawn -> 40 m east) this is exactly the route line.

  * Cross-track error is only accumulated during FLY, not during RTH, so the B-RID-style
    numbers stay comparable to a nominal run.

  * Moving obstacles need the clusterer to keep matching a fast thing from frame to frame. Its track_gate
    (default 0.3 m) allows only about 2.4 m/s of closing speed at 8 Hz, and the closing speed includes
    the drone's own 1.2 m/s. Run the clusterers with e.g.  -p track_gate:=0.8 -p process_hz:=10
"""

import csv
import json
import math
import os
from collections import deque
from datetime import datetime

import numpy as np
import rclpy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32, Float32MultiArray, String

# PX4 numbers for 'armed', 'offboard mode' and the commands we send to it.
# getattr = use the name from px4_msgs if it exists, otherwise use the plain number.
ARMED = getattr(VehicleStatus, "ARMING_STATE_ARMED", 2)
NAV_OFFBOARD = getattr(VehicleStatus, "NAVIGATION_STATE_OFFBOARD", 14)
CMD_ARM = getattr(VehicleCommand, "VEHICLE_CMD_COMPONENT_ARM_DISARM", 400)
CMD_MODE = getattr(VehicleCommand, "VEHICLE_CMD_DO_SET_MODE", 176)
CMD_LAND = getattr(VehicleCommand, "VEHICLE_CMD_NAV_LAND", 21)

# The four cameras, and the direction each one faces (radians: 0 = forward, positive = left).
CAMS = ("front", "left", "back", "right")
FACING = {"front": 0.0, "left": math.pi / 2, "back": math.pi, "right": -math.pi / 2}  # camera facing, body frame

# lat, lon, up (metres above the start point)
ROUTE = [
    (1.332504, 103.777035, 5.0),   # start (map centre), straight up to the fixed altitude of 5m
    (1.332504, 103.777394, 5.0),   # end, about 40 m east
]
# The route as one flat list of numbers: lat_1, lon_1, up, lat_2, lon_2, up, lat_n, lon_n, up, etc.
DEFAULT_WAYPOINTS = [v for point in ROUTE for v in point]

# Settings for fine tuning:
PARAMS = {
    "waypoints": DEFAULT_WAYPOINTS,
    "use_gps": True,

    # --- for flying ---
    "speed": 1.2,            # m/s along the line
    "accel": 0.5,            # m/s^2
    "decel": 1.7,            # m/s^2
    "accept_radius": 1.0,    # m
    "max_lead": 1.0,         # m, the carrot waits if the drone is further behind than this
    "land_at_end": True,
    "rate_hz": 20.0,

    # --- PX4 topics and logging ---
    "pos_topic": "/fmu/out/vehicle_local_position_v1",
    "status_topic": "/fmu/out/vehicle_status_v4",
    "csv_path": "",
    "mission_topic": "/dashboard/mission",   # status + finished route for the dashboard (drone_client.py forwards it)
    "flown_route_dir": "~",                  # the flown route is saved here as flown_route_<date>.json ("" = do not save)

    # --- cameras and their obstacle boxes ---
    "front_topic": "/obstacles/front/boxes",
    "left_topic": "/obstacles/left/boxes",
    "back_topic": "/obstacles/back/boxes",
    "right_topic": "/obstacles/right/boxes",
    "box_hold": 0.8,            # s, detections are kept this long so a flickering frame cannot erase an obstacle
    "obstacle_timeout": 1.5,    # s, a camera silent for this long counts as "not running"
    "require_front": True,      # hold still if the front camera is not publishing
    "allow_blind_dodge": False, # True: sidestep even if that side's camera is not running

    # --- slowing down and stopping for obstacles ---
    "slow_dist": 12.0,          # m, start slowing for an obstacle ahead
    "stop_dist": 5.0,           # m, stop here
    "body_half_width": 1.5,     # m, corridor half-width (drone radius + margin)
    "height_band": 1.5,         # m, obstacles higher or lower than this are ignored
    "clear_time": 2.0,          # s, must stay clear this long before moving on
    "pause_time": 2.0,          # s, wait this long before going round
    "give_up_time": 30.0,       # s, blocked this long -> recenter, then return home (RTH)
    
    # --- going round obstacles (a 'lane' = a sideways offset from the route line) ---
    "lane_step": 1.5,           # m between lanes
    "max_offset": 7.5,          # m, furthest lane from the line (route corridor is ~5 m)
    "lat_speed": 0.7,           # m/s sideways
    "pass_margin": 0.5,         # m, an obstacle must be this far behind before cutting back in
    "min_pass": 15.0,           # m, after starting a dodge, do not cut back to the line until this far along the route
    "left_bias": 0.2,           # m, tiny discount for going left. Only decides when left and right are about equal (0 = off)
    
    # --- remembering obstacles that have left the cameras' view ---
    "memory_time": 10.0,        # s, remember obstacles this long after they leave the cameras view (0 = off)
    "contradict_time": 2.0,     # s, forget a remembered obstacle if a camera can see its spot and sees nothing there
    "cam_fov_half_deg": 38.0,   # deg, each camera half field of view (a little under the real 43)
    "view_range": 8.0,          # m, how far a camera reliably sees
    
    # --- moving obstacles ---
    "mv_enabled": True,         # False = treat everything as standing still (like version 4)
    "mv_gate": 1.0,             # m, a new box this close to a known thing is the same thing
    "mv_window": 1.8,           # s, speed is worked out from this many seconds of recent positions (v7: 1.8, sparse boxes)
    "mv_speed_on": 0.6,         # m/s, faster than this for a moment = a moving obstacle
    "mv_speed_off": 0.35,       # m/s, slower than this for a while = standing still again
    "mv_min_updates": 3,        # times a thing must be seen before it can count as moving
    "mv_horizon": 4.0,          # s, how far ahead to predict
    "mv_body_half": 1.0,        # m, drone half-size used when predicting crashes with moving things (body_half_width is more cautious)
    "mv_margin": 0.5,           # m, extra room kept around a moving obstacle
    "mv_growth": 0.25,          # m per second, the extra room grows the further ahead we predict
    "mv_memory_time": 1.5,      # s, keep predicting a moving obstacle this long after the cameras lose it
    "mv_trail_time": 2.0,       # s, positions this recent are kept out of the static scene and the memory
    "mv_brake_room": 1.0,       # m, extra room kept in front of the drone when moving, so it can stop before the safety margin
    "mv_fuse_gap": 0.5,         # m, boxes from the cameras closer than this are one thing seen from two sides
    "mv_probe_time": 4.0,       # s, a newly seen thing gets this long to show whether it moves before the drone detours around it (v7: 4 s, a sweeping thing is slow at the ends of its sweep)
    "mv_retreat_speed": 1.0,    # m/s, speed of backing away along the route (standoff, or about to be hit and no lane is safe) (0 = off)
    "mv_search_speed": 1.5,     # m/s, a thing not seen for a moment may have moved this fast: widens the matching gate
    "standoff": 2.0,            # m, never hover closer than this to a standing obstacle ahead: back away
    "mv_standoff": 2.0,         # m, same for a moving obstacle / sway stripe: gap between the drone and its predicted box
    "mv_probe_hold": 15.0,      # s, a thing seen fewer than 3 times keeps blocking the way this long after it was last seen
    "mv_unconf_pad": 1.5,       # m, extra size (along the route) of the hold box around a thing not yet confirmed to stand still
    "mv_probe_lat": 7.5,        # m, extra size SIDEWAYS of that hold box: an unconfirmed thing this far off the line may be about to cross it
    "mv_end_slack": 1.0,        # m, extra room at a sway stripe end that we have seen the thing turn round at
    "mv_replan_gap": 1.0,       # s, least time between two changes of the sideways move because of a moving obstacle
    "mv_settle_time": 12.0,     # s, a sideways mover is treated as a no-go stripe for this long after it is first seen
    "mv_unknown_ext": 6.0,      # m, an end of a stripe we have not seen it turn round at is assumed to go on this far (v7: 6, was 60; 60 made every sweep wider than the cameras' reach unpassable. Raise it for more caution)
    "mv_sway_window": 20.0,     # s, for something that keeps turning round: everywhere it was this recently is no-go
    "mv_sway_memory": 30.0,     # s, keep a swaying (or new sideways) obstacle's no-go stripe this long after the cameras lose it
    "mv_sway_calm": 6.0,        # s, a swaying obstacle must be slow this long before it counts as standing still
    "mv_hidden_back": 0.5,      # m, extra size on every moving obstacle (the cameras only see its near side)
    "evade_lat_speed": 1.5,     # m/s, sideways speed when dodging a moving obstacle (lat_speed is slower)

    # --- lane planning and braking ---
    "plan_dist": 14.0,          # m, a lane must be clear this far ahead
    "brake_lead": 0.5,          # m, where the carrot stops relative to the drone when blocked
}


# --------------------------------------------------------------------------
# helpers (pure python/numpy)
# --------------------------------------------------------------------------
def gps_to_ned(lat, lon, up, ref_lat, ref_lon):
    """lat/lon (degrees) and height above start (m) -> PX4 local north, east, down (m)."""
    # Metres per degree of latitude / longitude at this spot on Earth.
    # (The Earth is not a perfect ball, so it is not just 111 km everywhere.)
    phi = math.radians(ref_lat)
    m_lat = 111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
    m_lon = 111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi)
    # north = latitude difference, east = longitude difference, down = minus the height
    return ((lat - ref_lat) * m_lat, (lon - ref_lon) * m_lon, -up)


def ned_to_gps(n, e, d, ref_lat, ref_lon):
    """The opposite of gps_to_ned: local north, east, down (m) -> lat, lon (degrees) and height above start (m)."""
    phi = math.radians(ref_lat)
    m_lat = 111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
    m_lon = 111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi)
    return ref_lat + n / m_lat, ref_lon + e / m_lon, -d


# Shortest distance from point p to the straight piece of line between a and b.
def point_segment_distance(p, a, b):
    ab = b - a
    denom = float(ab @ ab)
    # a and b are the same spot, so just measure to that spot
    if denom < 1e-9:
        return float(np.linalg.norm(p - a))
    # how far along a->b the closest point is (0 = at a, 1 = at b), kept within the piece
    t = float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def first_hit_x(boxes, x0, x1, y0, y1, zh):
    """Smallest forward distance at which any box overlaps the region
    x in [x0, x1], y (left) in [y0, y1], |z| < zh. inf if none. Boxes: (xlo,ylo,zlo,xhi,yhi,zhi)."""
    best = math.inf
    # go through every box and find the closest one that overlaps the region
    for lo_x, lo_y, lo_z, hi_x, hi_y, hi_z in boxes:
        # skip boxes that are completely outside the region
        if hi_x < x0 or lo_x > x1 or hi_y < y0 or lo_y > y1 or hi_z < -zh or lo_z > zh:
            continue
        # this box overlaps: keep the smallest forward distance (never less than the region's start)
        best = min(best, max(lo_x, x0))
    return best


def rotate_boxes(boxes, dpsi):
    """Re-express body-frame boxes in the route frame when the drone's heading differs from
    the route direction by dpsi (radians, heading minus route direction)."""
    # wrap the angle into the range -180..+180 degrees
    dpsi = (dpsi + math.pi) % (2 * math.pi) - math.pi
    # turned by less than about 1 degree (or no boxes): leave them as they are
    if abs(dpsi) < 0.02 or not boxes:
        return boxes
    c, s = math.cos(dpsi), math.sin(dpsi)
    out = []
    # turn each box: rotate its 4 corners, then draw a straight box around them
    # (a turned box is crooked, but the planner only wants straight boxes)
    for lo_x, lo_y, lo_z, hi_x, hi_y, hi_z in boxes:
        xs, ys = [], []
        for bx in (lo_x, hi_x):
            for by in (lo_y, hi_y):
                xs.append(bx * c + by * s)
                ys.append(-bx * s + by * c)
        out.append((min(xs), min(ys), lo_z, max(xs), max(ys), hi_z))
    return out


def choose_lane(boxes, cur, lanes, hw, zh, plan_dist, pass_margin, side_ok, left_bias=0.0, extra_ok=None):
    """Pick the cheapest acceptable lane. cur = the drone's sideways offset from the line
    (right +). Returns a lane offset, or None if there is no acceptable lane."""
    # try every lane and keep the cheapest one that is allowed
    best, best_cost = None, math.inf
    for lane in lanes:
        d = lane - cur           # sideways move needed, right +
        y_c = -d                 # same move in the body frame (left +)
        # if we have to move sideways to reach this lane, check that move first
        if abs(d) > 1e-6:
            # the camera on that side must be running
            if not side_ok(d):
                continue
            if first_hit_x(boxes, -pass_margin, hw, min(0.0, y_c) - hw, max(0.0, y_c) + hw, zh) < math.inf:
                continue         # something is in the way of the sideways move
        # the lane itself must be clear for plan_dist metres ahead
        if first_hit_x(boxes, 0.0, plan_dist, y_c - hw, y_c + hw, zh) < math.inf:
            continue             # the lane is blocked ahead
        # moving obstacles: the lane must also stay clear of where they are predicted to go
        if extra_ok is not None and not extra_ok(lane):
            continue
        # cheaper = smaller sideways move, and closer to the original line (the line counts 0.3x)
        cost = abs(d) + 0.3 * abs(lane)
        # Tiny discount for lanes on the LEFT of the line. This only picks between lanes that already passed
        # every safety check above, and it is too small to beat a real difference in cost. So it only decides
        # a true 50/50 (left and right equally good), and it also covers the drone sitting a few cm off the line.
        if lane < -1e-6:
            cost -= left_bias
        if cost < best_cost - 1e-9:
            best, best_cost = lane, cost
    return best


# True if two boxes overlap, or are within `pad` metres of touching.
def box_overlap(a, b, pad=0.5):
    return not (a[3] + pad < b[0] or b[3] + pad < a[0] or a[4] + pad < b[1] or b[4] + pad < a[1]
                or a[5] + pad < b[2] or b[5] + pad < a[2])


# Make one big box that contains all the given boxes.
def box_union(boxes):
    arr = np.array(boxes, float)
    return tuple(np.concatenate([arr[:, :3].min(axis=0), arr[:, 3:].max(axis=0)]))


def in_view(box, facings, half, rng):
    """True if the whole box (x forward, y left, relative to the drone) lies inside ONE camera's view wedge."""
    # the 4 floor corners of the box (only x and y matter here, not height)
    pts = [(box[i], box[j]) for i in (0, 3) for j in (1, 4)]
    # check each camera: all 4 corners must be close enough AND inside its view angle
    for phi in facings:
        if all(math.hypot(x, y) < rng and abs((math.atan2(y, x) - phi + math.pi) % (2 * math.pi) - math.pi) < half
               for x, y in pts):
            return True
    return False


# --------------------------------------------------------------------------
# moving obstacles (pure python/numpy, so they can be tested without ROS)
# --------------------------------------------------------------------------
# One thing the cameras keep seeing. Every incoming box is matched to a track like this one.
# From a track's recent positions we work out its speed; a track that keeps moving fast enough is a MOVING
# obstacle. Positions are in the PX4 world frame (north, east, up), NOT relative to the drone,
# so the drone's own movement does not look like the obstacle moving.
# A moving obstacle that turns round (swaying, circling, pacing) is marked BOUNDED: it cannot be predicted in a
# straight line, so we treat everywhere it has been lately as one fixed no-go zone instead.
class MTrack:
    _next_id = 1

    def __init__(self, det, now):
        self.id = MTrack._next_id
        MTrack._next_id += 1
        # det = (north_lo, east_lo, up_lo, north_hi, east_hi, up_hi)
        self.c = np.array([(det[0] + det[3]) / 2, (det[1] + det[4]) / 2])      # centre (north, east)
        self.half = np.array([(det[3] - det[0]) / 2, (det[4] - det[1]) / 2])   # half size (north, east)
        self.z0, self.z1 = det[2], det[5]                                        # height range (up)
        self.v = np.zeros(2)               # speed (north, east), m/s
        self.seen = now                    # last time a camera saw it
        self.born = now                    # first time a camera saw it
        self.samples = deque(maxlen=200)   # recent (time, north, east)
        self.samples.append((now, float(self.c[0]), float(self.c[1])))
        self.vref = deque(maxlen=80)       # recent (time, north speed, east speed) while it was moving fast
        self.n = 1                         # how many times it has been seen
        self.fast = 0                      # in a row: speed >= speed_on
        self.slow_t0 = None                # since when the speed has been below speed_off (None = it is not)
        self.moving = False
        self.bounded = False               # True once it has turned round
        self.cross = False                 # v7: True once it was seen moving ACROSS the route (set by the follower)


# Speed (north, east) from a straight line fitted through the recent positions.
# One noisy box cannot fool it: it needs several positions spread over some time. None = not enough yet.
def fit_velocity(samples, now, win, min_span=0.4, min_n=3):
    pts = [p for p in samples if now - p[0] <= win]
    # v7: with only two boxes (a sparse detector) use the straight line between them, but only if they are far
    # enough apart to be more than box noise and far enough apart in time
    if len(pts) == 2:
        dt2 = pts[1][0] - pts[0][0]
        d2 = math.hypot(pts[1][1] - pts[0][1], pts[1][2] - pts[0][2])
        if dt2 >= 0.25 and d2 >= 0.6:
            return np.array([(pts[1][1] - pts[0][1]) / dt2, (pts[1][2] - pts[0][2]) / dt2])
        return None
    if len(pts) < min_n or pts[-1][0] - pts[0][0] < min_span:
        return None
    t = np.array([p[0] for p in pts])
    t = t - t.mean()
    den = float((t * t).sum())
    if den < 1e-9:
        return None
    return np.array([float((t * np.array([p[1] for p in pts])).sum() / den),
                     float((t * np.array([p[2] for p in pts])).sum() / den)])


# Has the thing turned round? Compare how it moves now with how it moved 1 to 8 seconds ago.
# Pointing mostly the opposite way (both times clearly moving) = it turned round.
def _turned_round(t, now):
    sp = math.hypot(t.v[0], t.v[1])
    if sp < 0.4:
        return False
    for tt, a, b in t.vref:
        old = math.hypot(a, b)
        if 1.0 <= now - tt <= 8.0 and old >= 0.4 and a * t.v[0] + b * t.v[1] < -0.5 * old * sp:
            return True
    return False


# Update one track with a new box, and decide whether it counts as moving.
def _refresh_track(t, det, now, cfg):
    c = np.array([(det[0] + det[3]) / 2, (det[1] + det[4]) / 2])
    half = np.array([(det[3] - det[0]) / 2, (det[4] - det[1]) / 2])
    t.c = c
    t.half = 0.5 * t.half + 0.5 * half     # smooth the size so one odd frame does not jerk it around
    t.z0, t.z1 = det[2], det[5]
    t.seen = now
    t.n += 1
    # two cameras seeing it at the same moment count as one position
    if now - t.samples[-1][0] < 0.04:
        return
    t.samples.append((now, float(c[0]), float(c[1])))
    v = fit_velocity(t.samples, now, cfg["win"])
    if v is None:
        return
    t.v = 0.5 * t.v + 0.5 * v
    sp = float(np.hypot(t.v[0], t.v[1]))
    t.fast = t.fast + 1 if sp >= cfg["speed_on"] else 0
    if sp < cfg["speed_off"]:
        if t.slow_t0 is None:
            t.slow_t0 = now
    else:
        t.slow_t0 = None
    if not t.moving and t.n >= cfg["min_updates"] and t.fast >= 2:
        t.moving = True                    # fast for two checks in a row
    elif t.moving:
        # a swaying thing is slow for a moment at each end of its sway, so it must stay slow for much longer
        # before we believe it has really stopped
        calm = cfg["sway_calm"] if (t.bounded or t.cross) else cfg["calm"]
        if t.slow_t0 is not None and now - t.slow_t0 >= calm:
            t.moving, t.fast, t.bounded, t.cross = False, 0, False, False
        else:
            if sp >= 0.4:
                if not t.bounded and _turned_round(t, now):
                    t.bounded = True
                t.vref.append((now, float(t.v[0]), float(t.v[1])))


# The box (north_lo, east_lo, north_hi, east_hi) around all the places the centre of a track has been in the last win seconds.
def track_envelope(t, now, win):
    pts = [(n, e) for tt, n, e in t.samples if now - tt <= win] or [(float(t.c[0]), float(t.c[1]))]
    ns_, es_ = [p[0] for p in pts], [p[1] for p in pts]
    return min(ns_), min(es_), max(ns_), max(es_)


# Is the high end (top=True) or low end of s a real turning point? A real one is a bump in the middle of the
# record, within 0.4 m of the furthest reach, with the thing clearly lower on both sides of it.
def _end_known(s, top, prom=0.4):
    v = s if top else -s
    vmax = float(v.max())
    for k in range(1, len(v) - 1):
        if v[k] >= vmax - 0.4 and float(v[:k].min()) <= v[k] - prom and float(v[k + 1:].min()) <= v[k] - prom:
            return True
    return False


# Where a moving thing has been lately, along the line it moves on.
# Returns (centre, axis, s_lo, s_hi, lo_known, hi_known): the middle of its positions, the unit direction it
# moves along, how far its positions reach each way along that axis, and whether each end is a real turning
# point (seen turning round there) rather than just the place where we lost sight of it.
def track_extent(t, now, win):
    pts = [(n, e) for tt, n, e in t.samples if now - tt <= win]
    if len(pts) < 2:
        pts = [(float(t.c[0]), float(t.c[1]))] * 2
    P = np.array(pts)
    ctr = P.mean(axis=0)
    X = P - ctr
    if float(np.abs(X).max()) > 0.6 and len(P) >= 3:
        u = np.linalg.eigh(X.T @ X)[1][:, 1]           # the direction its positions spread out along the most
    else:
        sp = math.hypot(t.v[0], t.v[1])
        u = t.v / sp if sp > 1e-6 else np.array([0.0, 1.0])
    s = X @ u
    return ctr, u, float(s.min()), float(s.max()), _end_known(s, False), _end_known(s, True)


# Boxes that touch (or almost touch) were probably the SAME thing seen from different cameras: merge them into
# one box. dets = (north_lo, east_lo, up_lo, north_hi, east_hi, up_hi).
def fuse_dets(dets, gap=0.5):
    n = len(dets)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = dets[i], dets[j]
            if max(a[0], b[0]) - min(a[3], b[3]) <= gap and max(a[1], b[1]) - min(a[4], b[4]) <= gap:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(dets[i])
    return [(min(d[0] for d in g), min(d[1] for d in g), min(d[2] for d in g),
             max(d[3] for d in g), max(d[4] for d in g), max(d[5] for d in g)) for g in groups.values()]


# Match every new box (det) to the closest known track (or start a new track), then drop tracks not seen lately.
# cfg: gate (m), win (s), speed_on, speed_off (m/s), min_updates, calm, sway_calm, settle, keep_moving, keep_bounded, keep_static (s)
def update_mtracks(tracks, dets, now, cfg):
    cands = []
    for ti, t in enumerate(tracks):
        age = now - t.seen
        guess = t.c + (t.v * age if t.moving and not t.bounded else 0.0)   # a thing on a straight run is expected to have moved on
        for di, d in enumerate(dets):
            dist = math.hypot((d[0] + d[3]) / 2 - guess[0], (d[1] + d[4]) / 2 - guess[1])
            # a moving thing that was out of sight for a moment may have moved on by about its speed times that time
            reach = cfg["gate"] + ((math.hypot(t.v[0], t.v[1]) + 0.5) if t.moving else cfg.get("search_speed", 0.0)) * min(age, 2.0)
            if dist < reach:
                cands.append((dist, ti, di))
    cands.sort()
    used_t, used_d = set(), set()
    for _, ti, di in cands:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        _refresh_track(tracks[ti], dets[di], now, cfg)
    # v7: a box that appears inside the area a crossing / swaying thing has been sweeping is that thing again
    for di, d in enumerate(dets):
        if di in used_d:
            continue
        cx_, cy_ = (d[0] + d[3]) / 2, (d[1] + d[4]) / 2
        best_t, best_d = None, 1.5
        for ti, t in enumerate(tracks):
            if ti in used_t or not (t.cross or t.bounded) or now - t.seen > cfg["keep_bounded"]:
                continue
            n0, e0, n1, e1 = track_envelope(t, now, cfg.get("sway_window", 20.0))
            dist = math.hypot(max(n0 - cx_, 0.0, cx_ - n1), max(e0 - cy_, 0.0, cy_ - e1))
            if dist < best_d:
                best_t, best_d = ti, dist
        if best_t is not None:
            used_t.add(best_t)
            used_d.add(di)
            _refresh_track(tracks[best_t], d, now, cfg)
    for di, d in enumerate(dets):
        if di not in used_d:
            tracks.append(MTrack(d, now))
    # a moving thing that turns round, crosses the route, or was first seen lately, is kept longer: it may come back
    tracks[:] = [t for t in tracks if now - t.seen <= (
        cfg["keep_bounded"] if t.moving and (t.bounded or t.cross or now - t.born < cfg["settle"]) else
        cfg["keep_moving"] if t.moving else cfg["keep_unconf"] if t.n < 3 else cfg["keep_static"])]
    return tracks


# Time (s) of the first predicted crash between the drone and any moving obstacle, or None.
# Everything is relative to the drone, route frame: x forward, y left, the drone starts at (0, 0).
# preds = (cx, cy, half_x, half_y, z_lo, z_hi, vx, vy, extra_room, growth) per moving obstacle.
# The drone flies forward at v_fwd and slides sideways towards y_goal at lat_v. A crash = the drone's
# square (half-width hw) overlaps the obstacle's box plus extra_room, which grows by `growth` m per second.
# ahead   = extra room in FRONT of the drone while it moves forward (it needs room to stop).
# v_after = if given, the drone first slides to y_goal (not moving forward, at v_fwd), and THEN flies forward at
#           v_after. This is how a detour really goes, and it also checks that the new lane lets us get on.
#           Then we look `horizon` seconds past the end of the slide. Only an obstacle that stands in one place
#           (a no-go zone: speed 0 and no growth) is checked that far ahead; a moving one cannot be predicted that far.
# slack   = extra room on top, used when CHOOSING a lane, so that a lane that is only just clear is not picked
#           (the zone may turn out a little bigger than we think).
# soft_slide = while the drone is still sliding away from a no-go zone, only real contact counts (not the extra room).
#           This lets it escape from a spot where it has ended up a little too close.
def first_conflict(preds, y_goal, v_fwd, lat_v, hw, zh, horizon=4.0, dt=0.2, ahead=0.0, v_after=None,
                   slack=0.0, soft_slide=False):
    t_slide = abs(y_goal) / lat_v if lat_v > 0 else 0.0
    steps = int((t_slide + horizon) / dt) + 1 if v_after is not None else int(horizon / dt) + 1
    for k in range(steps):
        t = k * dt
        y = math.copysign(min(abs(y_goal), lat_v * t), y_goal) if y_goal else 0.0
        if v_after is None:
            x, going = v_fwd * t, v_fwd > 0
        else:
            x, going = v_fwd * min(t, t_slide) + v_after * max(0.0, t - t_slide), v_after > 0 and t > t_slide
        for cx, cy, hx, hy, z0, z1, vx, vy, pad0, growth in preds:
            if z1 < -zh or z0 > zh:
                continue                                    # not at our height
            fixed = not (vx or vy or growth)                # a no-go zone: stays where it is
            if t > horizon and not fixed:
                continue                                    # a moving obstacle cannot be predicted this far ahead
            pad = pad0 + growth * t + slack
            if soft_slide and fixed and t <= t_slide:
                pad = 0.0
            if fixed and going and cx + hx < 0.0:
                pad = 0.0           # the zone is behind us and we are moving away from it: only real contact counts
            if abs(x - (cx + vx * t)) < hx + pad + hw + (ahead if going and cx + vx * t > x else 0.0) and abs(y - (cy + vy * t)) < hy + pad + hw:
                return t
    return None


# How close the drone gets to a moving obstacle (no extra room): the smallest gap in metres over the horizon.
# Negative = they touch. Used to take the least bad lane when no lane is completely clear.
def min_gap(preds, y_goal, v_fwd, lat_v, hw, zh, horizon=4.0, dt=0.2):
    best = math.inf
    for k in range(int(horizon / dt) + 1):
        t = k * dt
        y = math.copysign(min(abs(y_goal), lat_v * t), y_goal) if y_goal else 0.0
        x = v_fwd * t
        for cx, cy, hx, hy, z0, z1, vx, vy, pad0, growth in preds:
            if z1 < -zh or z0 > zh:
                continue
            best = min(best, max(abs(x - (cx + vx * t)) - hx - hw, abs(y - (cy + vy * t)) - hy - hw))
    return best


# Try forward speeds 100%, 75%, 50%, 25%, 0% (while sliding to y_goal) and keep the fastest one with no
# predicted crash. Returns (scale, urgent). urgent = even 0% gets hit, so we must dodge right now.
# `ahead` = extra room kept in front of the drone whenever it moves, so that it can stop before the safety margin.
def plan_vs_movers(preds, y_goal, v_nom, lat_v, hw, zh, horizon=4.0, ahead=0.0):
    for sc in (1.0, 0.75, 0.5, 0.25, 0.0):
        if first_conflict(preds, y_goal, v_nom * sc, lat_v, hw, zh, horizon, ahead=ahead) is None:
            return sc, False
    return 0.0, True


# --------------------------------------------------------------------------
# ros2 node
# --------------------------------------------------------------------------
# The whole drone controller. on_timer() runs 20 times a second and does the work for
# whichever state we are in. Normal order of states:
    # WAIT -> STREAM -> ENGAGE -> FLY -> LAND (or HOLD) -> DONE

# RECENTER and RTH (return to home) only happen if the drone gets stuck.
class WaypointFollower(Node):
    def __init__(self):
        super().__init__("waypoint_follower")
        # declare every setting from PARAMS, then read the values into cfg
        free = ParameterDescriptor(dynamic_typing=True)
        for name, default in PARAMS.items():
            self.declare_parameter(name, default, free)
        cfg = {name: self.get_parameter(name).value for name in PARAMS}
        self.cfg = cfg

        # the waypoints arrive as one flat list: turn it into rows of three numbers
        raw = np.array(cfg["waypoints"], dtype=float).ravel()
        if raw.size == 0 or raw.size % 3:
            raise ValueError("waypoints must be a flat list of triples")
        wp = raw.reshape(-1, 3)
        # With GPS: waypoints are converted later, once PX4 tells us where (0, 0) is on the map.
        # Without GPS: they are already in local metres (north, east, up), so just flip 'up' into 'down'.
        self.use_gps = bool(cfg["use_gps"])
        self.wp_raw = wp
        self.wps = None if self.use_gps else np.column_stack([wp[:, 0], wp[:, 1], -wp[:, 2]])
        self.ref = None

        # copy each setting into its own variable (float() makes sure it is a number)
        f = lambda k: float(cfg[k])
        self.speed, self.accel, self.decel = f("speed"), f("accel"), f("decel")
        self.accept_radius, self.max_lead = f("accept_radius"), f("max_lead")
        self.land_at_end = bool(cfg["land_at_end"])
        self.rate_hz = f("rate_hz")
        self.obstacle_timeout = f("obstacle_timeout")
        self.box_hold = f("box_hold")
        self.require_front = bool(cfg["require_front"])
        self.allow_blind = bool(cfg["allow_blind_dodge"])
        self.slow_dist, self.stop_dist = f("slow_dist"), f("stop_dist")
        self.hw, self.zh = f("body_half_width"), f("height_band")
        self.clear_time, self.pause_time, self.give_up_time = f("clear_time"), f("pause_time"), f("give_up_time")
        self.lane_step, self.max_offset = f("lane_step"), f("max_offset")
        self.lat_speed, self.pass_margin = f("lat_speed"), f("pass_margin")
        self.plan_dist, self.brake_lead = f("plan_dist"), f("brake_lead")
        self.min_pass, self.left_bias = f("min_pass"), f("left_bias")
        self.memory_time, self.contradict_time = f("memory_time"), f("contradict_time")
        self.fov_half, self.view_range = math.radians(f("cam_fov_half_deg")), f("view_range")
        self.mem = []           # remembered obstacles: (box in route coordinates, last time seen)
        # moving obstacles
        self.mv_on = bool(cfg["mv_enabled"])
        self.mv_horizon, self.mv_margin = f("mv_horizon"), f("mv_margin")
        self.mv_hw = f("mv_body_half")
        self.mv_growth, self.mv_memory_time = f("mv_growth"), f("mv_memory_time")
        self.mv_trail_time, self.evade_lat_speed = f("mv_trail_time"), f("evade_lat_speed")
        self.mv_sway_window, self.mv_sway_memory = f("mv_sway_window"), f("mv_sway_memory")
        self.mv_hidden = f("mv_hidden_back")
        self.mv_brake = f("mv_brake_room")
        self.mv_retreat, self.mv_replan_gap = f("mv_retreat_speed"), f("mv_replan_gap")
        self.mv_fuse_gap, self.mv_probe = f("mv_fuse_gap"), f("mv_probe_time")
        self.standoff, self.mv_standoff = f("standoff"), f("mv_standoff")
        self.lead_now = f("brake_lead")
        self.mv_end_slack = f("mv_end_slack")
        self.mv_probe_hold, self.mv_unconf_pad = f("mv_probe_hold"), f("mv_unconf_pad")
        self.mv_probe_lat = f("mv_probe_lat")
        self.accel_now = self.accel
        self.det_log_t = 0.0
        self.det_buf = []                 # boxes from the cameras (world frame) waiting to be merged and tracked
        self.det_count = {c: 0 for c in CAMS}   # how many box messages each camera gave, for the log
        self.mv_replan_t = -1e9     # last time a sideways move was changed because of a moving obstacle
        self.mv_settle, self.mv_unknown_ext = f("mv_settle_time"), f("mv_unknown_ext")
        self.mv_cfg = dict(gate=f("mv_gate"), win=f("mv_window"), speed_on=f("mv_speed_on"),
                           speed_off=f("mv_speed_off"), min_updates=int(f("mv_min_updates")),
                           calm=0.6, sway_calm=f("mv_sway_calm"), settle=f("mv_settle_time"),
                           keep_moving=self.mv_memory_time + 0.5, keep_bounded=self.mv_sway_memory + 0.5, keep_static=3.0, keep_unconf=f("mv_probe_hold"), probe=f("mv_probe_time"),
                           search_speed=f("mv_search_speed"), sway_window=f("mv_sway_window"))
        self.mtracks = []       # everything the cameras have seen lately, see MTrack
        self.evading = False    # True while sliding sideways out of a moving obstacle's way
        self.mv_seen = set()    # ids of moving obstacles already announced in the log
        self.mv_swaying = set() # ids of obstacles already announced as 'keeps turning round'
        # the allowed lanes (sideways offsets from the line): 0, then 1 step right, 1 step left,
        # 2 steps right, 2 steps left, ... up to max_offset
        n = int(self.max_offset // self.lane_step)
        self.lanes = [0.0] + [s * k * self.lane_step for k in range(1, n + 1) for s in (1.0, -1.0)]

        # state
        self.state = "WAIT"
        # latest info from PX4: position (north, east, down), which way it faces, and if it can be trusted
        self.pos = None
        self.heading = 0.0
        self.pos_valid = False
        # PX4 status: armed? in offboard mode? has any status message arrived yet?
        self.armed = False
        self.offboard = False
        self.have_status = False
        # start-up bookkeeping (setpoints sent so far, time of the last command, log timers, ...)
        self.stream_count = 0
        self.last_cmd_t = 0.0
        self._log_t = {}
        self.t_last = None
        self.engage_t0 = None
        self.warned_offboard = False
        # hold_pos / hold_yaw = where to stay and which way to face when holding still.
        # home_pos = where RTH flies back to (saved when the first waypoint is reached).
        self.hold_pos = None
        self.hold_yaw = 0.0
        self.home_pos = None
        self.home_yaw = 0.0
        # The 'carrot' is the point we tell PX4 to fly to. It slides along the route line and the drone chases it.
        # seg = which leg of the route we are on, seg_start = where that leg began,
        # carrot_s = how far along the leg the carrot is, v = the carrot's speed
        self.seg = 0
        self.seg_start = None
        self.carrot_s = 0.0
        self.v = 0.0
        self.fly_t0 = None
        self.off_cmd = 0.0      # sideways offset being commanded (right +)
        self.off_goal = 0.0
        self.dodge_side = 0.0   # +1 right / -1 left while going round something
        self.dodge_s = 0.0      # along-route position where this dodge started
        self.leg_done = False   # carrot has reached the end of the current leg
        # block_t0 = when we first got blocked (None = not blocked)
        # last_blocked_t = the last time something was in the way
        self.block_t0 = None
        self.last_blocked_t = -1e9
        # cross-track error (how far we are from the route line): running totals for the summary at the end
        self.xte_sum = self.xte_sq = self.xte_max = 0.0
        self.xte_n = 0
        self.hist = {c: [] for c in CAMS}   # (time, boxes, drone position) of recent messages per camera
        # when each camera last sent anything (used to tell whether it is running)
        self.box_t = {c: -1e9 for c in CAMS}
        # distance to the nearest obstacle ahead (inf = nothing ahead)
        self.fx = math.inf

        # for the dashboard
        self.outcome = None            # how the mission ended: "arrived", "returned" or "aborted" (None = still going)
        self.giveup_reason = ""        # why it decided to go home
        self.returning = False         # True once it has decided to go home
        self.path = []                 # flown path: (north, east, down, leg), leg 0 = outbound, 1 = on the way home
        self.mission_t = -1e9          # last time the status was published
        self.complete_msg = None       # the finished route as JSON text, published every 5 s once it exists
        self.complete_t = -1e9
        self.land_t0 = None

        # optional: log the flight to a CSV file if csv_path is set
        self.csv_file = self.csv = None
        if cfg["csv_path"]:
            self.csv_file = open(cfg["csv_path"], "w", newline="")
            self.csv = csv.writer(self.csv_file)
            self.csv.writerow(["t", "x_n", "y_e", "z_d", "carrot_n", "carrot_e", "carrot_d",
                               "cross_track", "offset_cmd", "front_clear_m", "target_index"])

        # message delivery settings that PX4 expects (best effort = fast, may drop a message now and then)
        pub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        sub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        # publishers: the offboard heartbeat, the position target, commands (arm / land), and the cross-track error
        self.mode_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", pub_qos)
        self.sp_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", pub_qos)
        self.cmd_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", pub_qos)
        self.xte_pub = self.create_publisher(Float32, "/route/cross_track_error", 10)
        self.mission_pub = self.create_publisher(String, cfg["mission_topic"], 10)
        # subscribers: PX4 position, PX4 status, and the four obstacle-box topics
        self.create_subscription(VehicleLocalPosition, cfg["pos_topic"], self.on_pos, sub_qos)
        self.create_subscription(VehicleStatus, cfg["status_topic"], self.on_status, sub_qos)
        # one listener per camera (each one stores into self.hist and self.box_t)
        self.cb = {}
        for cam in CAMS:
            self.cb[cam] = self._make_box_cb(cam)
            self.create_subscription(Float32MultiArray, cfg[f"{cam}_topic"], self.cb[cam], 10)
        # run on_timer() rate_hz times a second: this is the main loop
        self.create_timer(1.0 / self.rate_hz, self.on_timer)
        self.get_logger().info(
            f"{len(wp)} waypoints, speed {self.speed} m/s. Obstacle boxes from {[cfg[c + '_topic'] for c in CAMS]}")

    # ------------------------------------------------------------------ inputs
    def on_pos(self, msg):
        # PX4 sends position (north, east, down) and heading. 'valid' = PX4 trusts its own position.
        self.pos = np.array([msg.x, msg.y, msg.z], dtype=float)
        self.heading = float(msg.heading)
        self.pos_valid = bool(getattr(msg, "xy_valid", True)) and bool(getattr(msg, "z_valid", True))
        # PX4 also tells us which lat/lon is (0, 0); we need that to convert GPS waypoints
        if self.use_gps and getattr(msg, "xy_global", False):
            self.ref = (float(msg.ref_lat), float(msg.ref_lon))

    def on_status(self, msg):
        # remember if the drone is armed and if PX4 is in offboard mode (our commands only work in offboard)
        self.armed = msg.arming_state == ARMED
        self.offboard = msg.nav_state == NAV_OFFBOARD
        self.have_status = True

    # Makes the listener for ONE camera's boxes (one listener per camera, so each knows its own name).
    def _make_box_cb(self, cam):
        def cb(msg):
            # the message is a flat list with 6 numbers per box: split it into boxes
            d = np.asarray(msg.data, dtype=float)
            boxes = [tuple(d[i:i + 6]) for i in range(0, len(d) - 5, 6)]
            # note the time, and where the drone was, so the boxes can be shifted later as the drone moves
            now = self._now()
            self.box_t[cam] = now
            p0 = None if self.pos is None else self.pos.copy()   # where the drone was when these boxes were measured
            # keep only the last box_hold seconds of messages, plus this new one
            self.hist[cam] = [h for h in self.hist[cam] if now - h[0] < self.box_hold] + [(now, boxes, p0)]
            # moving obstacles: match every box to something seen before, and work out its speed
            if self.mv_on and p0 is not None and boxes:
                self.det_count[cam] += 1
                self._track_boxes(now, boxes, p0, self.heading)
        return cb

    # ------------------------------------------------------------------ outputs
    # current time in seconds
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # current time in microseconds (what PX4 messages want)
    def _stamp_us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    # PX4 only accepts our control if we keep saying 'I'm controlling the position' many times a second.
    # This 'heartbeat' must never stop while flying.
    def publish_heartbeat(self):
        m = OffboardControlMode()
        m.timestamp = self._stamp_us()
        m.position = True
        self.mode_pub.publish(m)

    # Tell PX4 where to fly (north, east, down) and which way to face.
    def publish_setpoint(self, pos_ned, yaw):
        m = TrajectorySetpoint()
        m.timestamp = self._stamp_us()
        m.position = [float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2])]
        m.yaw = float(yaw)
        # set everything except position and yaw to NaN = 'ignore this', so only position + yaw are used
        for field in ("velocity", "acceleration", "jerk"):
            if hasattr(m, field):
                setattr(m, field, [math.nan] * 3)
        if hasattr(m, "yawspeed"):
            m.yawspeed = math.nan
        self.sp_pub.publish(m)

    # Send a command to PX4 (arm, change mode, land). Target 1 / 1 = the flight controller.
    def send_command(self, command, p1=0.0, p2=0.0):
        m = VehicleCommand()
        m.timestamp = self._stamp_us()
        m.command = int(command)
        m.param1 = float(p1)
        m.param2 = float(p2)
        m.target_system = 1
        m.target_component = 1
        m.source_system = 1
        m.source_component = 1
        m.from_external = True
        self.cmd_pub.publish(m)

    # ------------------------------------------------------------------ helpers
    # Change state and print it in the log.
    def _set_state(self, new):
        self.get_logger().info(f"State: {self.state} -> {new}")
        if new == "LAND" and self.outcome is None:
            self.outcome = "aborted"      # landing without having finished the route or got home
        self.state = new

    # Print a message at most once every `seconds`, so the screen does not fill up.
    def _log_every(self, now, seconds, text):
        key = text[:24]   # each kind of message has its own timer
        if now - self._log_t.get(key, -1e9) >= seconds:
            self._log_t[key] = now
            self.get_logger().info(text)

    # A camera is 'alive' if it sent something recently.
    def _alive(self, cam, now):
        return now - self.box_t[cam] < self.obstacle_timeout

    # Returns a function: side_ok(d) = may we move that way? (d > 0 = right).
    # It needs that side's camera running, unless allow_blind_dodge is on.
    def _side_ok(self, now):
        left = self._alive("left", now) or self.allow_blind
        right = self._alive("right", now) or self.allow_blind
        return lambda d: right if d > 0 else left

    # Builds a message that explains why no lane was found (for the log).
    def _why_no_lane(self, now):
        missing = [c for c in ("left", "right") if not self._alive(c, now)]
        if missing and not self.allow_blind:
            return (f"Side camera(s) not publishing: {', '.join(missing)}. Start their clusterers "
                    f"(-p ns:=left / -p ns:=right), or use -p allow_blind_dodge:=true (test only).")
        if self.dodge_side:
            return (f"No lane on dodge side ({'right' if self.dodge_side > 0 else 'left'}). "
                    f"Widen max_offset, reduce pass_margin, or increase plan_dist.")
        return ("Every lane within max_offset is blocked ahead or beside it. Try a larger -p max_offset "
                "(your route corridor is about 5 m) or a lower -p pass_margin.")
    
    def _remember(self, now, live, s_drone, cross, excl=()):
        """Merge what the cameras see into a memory kept in ROUTE coordinates (s along the route, l to the left,
        absolute height), so obstacles that fall out of view are not mistaken for free space. Returns live + remembered
        boxes, relative to the drone."""
        # memory is turned off: just use what the cameras see right now
        if self.memory_time <= 0:
            return live
        # where the drone is in route coordinates: l_d = sideways (left +), alt = height above the start
        l_d, alt = -cross, -float(self.pos[2])
        # 1) Add what the cameras see now: convert each box to route coordinates, merge it with any
        # remembered box it overlaps, and stamp it with the current time.
        for b in live:
            rb = (b[0] + s_drone, b[1] + l_d, b[2] + alt, b[3] + s_drone, b[4] + l_d, b[5] + alt)
            hits = [m for m in self.mem if box_overlap(m[0], rb)]
            self.mem = [m for m in self.mem if m not in hits]
            self.mem.append((box_union([rb] + [m[0] for m in hits]), now))   # grows as more of it is seen
        # 2) Throw away memories that are too old, too far behind, or contradicted by a camera.
        # (facings = the directions of the cameras that are running)
        facings = [FACING[c] for c in CAMS if self._alive(c, now)]
        keep = []
        for box, t in self.mem:
            # this remembered box, relative to the drone as it is now
            rel = (box[0] - s_drone, box[1] - l_d, box[2] - alt, box[3] - s_drone, box[4] - l_d, box[5] - alt)
            if excl and self._inside_any(rel, excl):
                continue                                    # a moving thing was here: it is not a standing obstacle
            if now - t > self.memory_time or rel[3] < -30.0:
                continue                                    # too old, or far behind us
            if now - t > self.contradict_time:              # a camera can see the nearest part of it, and nothing is there
                nx, ny = min(max(0.0, rel[0]), rel[3]), min(max(0.0, rel[1]), rel[4])
                if in_view((nx, ny, 0.0, nx, ny, 0.0), facings, self.fov_half, self.view_range):
                    continue
            keep.append((box, t))
        self.mem = keep
        # give back the live boxes + the remembered ones, all relative to the drone again
        return live + [(b[0] - s_drone, b[1] - l_d, b[2] - alt, b[3] - s_drone, b[4] - l_d, b[5] - alt)
                       for b, t in self.mem]

    # Collect all the boxes the running cameras reported in the last box_hold seconds,
    # shifted for how far the drone has moved since, and turned to match the route direction.
    def _scene(self, now):
        c, s = math.cos(self.hold_yaw), math.sin(self.hold_yaw)
        boxes = []
        for cam in CAMS:
            # ignore cameras that have gone quiet
            if not self._alive(cam, now):
                continue
            for t, bl, p0 in self.hist[cam]:
                # ignore messages that are too old
                if now - t >= self.box_hold:
                    continue
                dx = dy = 0.0
                if p0 is not None and self.pos is not None:   # how far the drone has moved since the measurement
                    d = self.pos - p0
                    dx, dy = d[0] * c + d[1] * s, d[0] * s - d[1] * c
                # boxes were measured from where the drone was then: subtract its movement since,
                # so they are relative to where it is now
                boxes.extend((b[0] - dx, b[1] - dy, b[2], b[3] - dx, b[4] - dy, b[5]) for b in bl)
        # if the drone has turned away from the route direction, turn the boxes too
        return rotate_boxes(boxes, self.heading - self.hold_yaw)

    # ------------------------------------------------------------------ moving obstacles
    # Turn boxes from a camera (drone body frame) into world-frame boxes (north, east, up) and keep them until the
    # next tick, when boxes from all cameras are merged and matched to tracks (see _track_update).
    def _track_boxes(self, now, boxes, p0, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        alt = -float(p0[2])
        for b in boxes:
            # the 4 floor corners of the box, body frame (x forward, y left) -> north / east
            corners = [(b[0], b[1]), (b[0], b[4]), (b[3], b[1]), (b[3], b[4])]
            ns = [p0[0] + x * c + y * s for x, y in corners]
            es = [p0[1] + x * s - y * c for x, y in corners]
            self.det_buf.append((min(ns), min(es), alt + b[2], max(ns), max(es), alt + b[5]))

    # Merge the boxes that arrived since the last tick (all cameras), match them to tracks and work out speeds.
    def _track_update(self, now):
        dets, self.det_buf = fuse_dets(self.det_buf, self.mv_fuse_gap), []
        before = {t.id for t in self.mtracks if t.moving}
        if dets:
            update_mtracks(self.mtracks, dets, now, self.mv_cfg)
        for t in self.mtracks:
            if t.moving and t.id not in before and t.id not in self.mv_seen:
                self.mv_seen.add(t.id)
                rng_m = float(np.hypot(t.c[0] - self.pos[0], t.c[1] - self.pos[1])) if self.pos is not None else float("nan")
                self.get_logger().info(f"Moving obstacle #{t.id}: {float(np.hypot(*t.v)):.1f} m/s "
                                       f"(north {t.v[0]:+.1f}, east {t.v[1]:+.1f}), {rng_m:.1f} m away, "
                                       f"first seen {now - t.born:.1f} s ago.")
            if t.bounded and t.id not in self.mv_swaying:
                self.mv_swaying.add(t.id)
                n0, e0, n1, e1 = track_envelope(t, now, self.mv_sway_window)
                self.get_logger().info(f"Obstacle #{t.id} keeps turning round: treating its whole sway area as no-go "
                                       f"({n1 - n0:.1f} m north-south, {e1 - e0:.1f} m east-west so far).")
        # how often did each camera give us boxes? (dropouts here are why a thing gets lost)
        if now - self.det_log_t >= 5.0:
            self.det_log_t = now
            self.get_logger().info("Detections in the last 5 s (box messages with something in them): "
                                   + ", ".join(f"{c} {n}" for c, n in self.det_count.items())
                                   + f"; tracks now: {len(self.mtracks)} ({sum(t.moving for t in self.mtracks)} moving).")
            self.det_count = {c: 0 for c in CAMS}

    # v7: seen, but not yet confirmed to stand still: fewer than 3 sightings, or younger than mv_probe_time. Such a thing may
    # be a sweeping obstacle that is only seen once per sweep, so no detour is planned around it, and it keeps blocking
    # the way (as an inflated hold box) for mv_probe_hold seconds after it was last seen (3 s once it has 3 sightings).
    def _unconfirmed(self, t, now):
        return (not t.moving) and (t.n < 3 or now - t.born < self.mv_probe) and \
               now - t.seen < (self.mv_probe_hold if t.n < 3 else 3.0)

    # Is there a newly seen thing that has not yet shown whether it moves?
    def _probing(self, now):
        return any(self._unconfirmed(t, now) for t in self.mtracks)

    # Hold boxes (route frame, relative to the drone) for the things that are not yet confirmed to stand still.
    def _hold_boxes(self, now):
        if self.pos is None:
            return []
        c, s = math.cos(self.hold_yaw), math.sin(self.hold_yaw)
        alt = -float(self.pos[2])
        out = []
        for t in self.mtracks:
            if not self._unconfirmed(t, now):
                continue
            dn, de = t.c[0] - self.pos[0], t.c[1] - self.pos[1]
            x, y = dn * c + de * s, dn * s - de * c
            hx = abs(c) * t.half[0] + abs(s) * t.half[1] + self.mv_unconf_pad
            hy = abs(s) * t.half[0] + abs(c) * t.half[1] + self.mv_probe_lat
            out.append((x - hx, y - hy, t.z0 - alt, x + hx, y + hy, t.z1 - alt))
        return out

    # Moving obstacles as predictions in the route frame, relative to the drone (x forward, y left).
    # preds = (cx, cy, half_x, half_y, z_lo, z_hi, vx, vy, extra_room, growth), one per moving obstacle.
    #   - a thing on a straight run (or heading towards us): its box is carried forward at its speed
    #   - a sideways mover that is new, or that keeps turning round: a fixed no-go STRIPE along the line it moves on.
    #     An end we have not seen it turn round at is assumed to go on for mv_unknown_ext metres.
    # excl  = boxes (x_lo, y_lo, x_hi, y_hi) where a moving thing is or has been lately (only the part we really saw).
    #         Standing-obstacle boxes centred inside these are ignored, so a moving thing does not leave fake
    #         obstacles behind it.
    def _movers(self, now):
        if self.pos is None:
            return [], []
        c, s = math.cos(self.hold_yaw), math.sin(self.hold_yaw)
        alt = -float(self.pos[2])
        self.mtracks = [t for t in self.mtracks if now - t.seen <= (
            self.mv_sway_memory + 0.5 if t.moving and (t.bounded or t.cross or now - t.born < self.mv_settle) else
            self.mv_memory_time + 0.5 if t.moving else self.mv_probe_hold if t.n < 3 else 3.0)]
        rel = lambda n, e: ((n - self.pos[0]) * c + (e - self.pos[1]) * s, (n - self.pos[0]) * s - (e - self.pos[1]) * c)
        preds, excl = [], []
        for t in self.mtracks:
            age = now - t.seen
            if not t.moving:
                continue
            # half size, plus a little extra: the cameras only see the near side, so the far side may stick out
            hn, he = t.half[0] + self.mv_hidden, t.half[1] + self.mv_hidden
            vx, vy = t.v[0] * c + t.v[1] * s, t.v[0] * s - t.v[1] * c
            sideways = abs(vy) >= 0.5 * math.hypot(vx, vy)
            if sideways:
                t.cross = True          # v7: remembered as a sway stripe from now on
            if t.bounded or t.cross:
                if age > self.mv_sway_memory:
                    continue
                ctr, u, s_lo, s_hi, lo_k, hi_k = track_extent(t, now, self.mv_sway_window)
                w = max(hn, he)
                perp = np.array([-u[1], u[0]])

                def box(a, b):
                    # the stripe from a to b along the axis, w wide each side, as a straight box in the route frame
                    pts = [rel(*(ctr + u * sa + perp * sw)) for sa in (a, b) for sw in (-w, w)]
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    return min(xs), min(ys), max(xs), max(ys)

                sl = self.mv_end_slack
                x0, y0, x1, y1 = box(s_lo - sl if lo_k else s_lo - self.mv_unknown_ext,
                                     s_hi + sl if hi_k else s_hi + self.mv_unknown_ext)
                preds.append(((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2,
                              t.z0 - alt, t.z1 - alt, 0.0, 0.0, self.mv_margin, 0.0))
                ox0, oy0, ox1, oy1 = box(s_lo, s_hi)            # only what we really saw
                excl.append((ox0 - 0.5, oy0 - 0.5, ox1 + 0.5, oy1 + 0.5))
                continue
            if age > self.mv_memory_time:
                continue
            # half size in the route frame (a turned box is crooked: use the straight box around it)
            hx = abs(c) * hn + abs(s) * he
            hy = abs(s) * hn + abs(c) * he
            # where it should be NOW: last seen position + speed * time since (dead reckoning)
            x, y = rel(t.c[0] + t.v[0] * age, t.c[1] + t.v[1] * age)
            # the longer ago it was seen, the less sure we are, so keep more room
            preds.append((x, y, hx, hy, t.z0 - alt, t.z1 - alt, vx, vy,
                          self.mv_margin + self.mv_growth * age, self.mv_growth))
            excl.append((x - hx - 0.5, y - hy - 0.5, x + hx + 0.5, y + hy + 0.5))
            for tt, n, e in list(t.samples):
                if now - tt <= self.mv_trail_time:
                    px, py = rel(n, e)
                    excl.append((px - hx - 0.5, py - hy - 0.5, px + hx + 0.5, py + hy + 0.5))
        return preds, excl

    # Is the centre of box b (x_lo, y_lo, z_lo, x_hi, y_hi, z_hi) inside any of the boxes in excl?
    @staticmethod
    def _inside_any(b, excl):
        cx, cy = (b[0] + b[3]) / 2, (b[1] + b[4]) / 2
        return any(e[0] <= cx <= e[2] and e[1] <= cy <= e[3] for e in excl)

    def _avoid(self, now, dt, cross, s_drone):
        """Decide forward speed (0..1) and update the sideways offset. Returns (scale, land)."""
        self.fx = math.inf
        # no front camera data: either stand still (safe) or fly on blind, depending on require_front
        if not self._alive("front", now):
            if self.require_front:
                self._log_every(now, 5.0, "No front obstacle data: holding. (Start the front clusterer, "
                                          "or run with -p require_front:=false)")
                return 0.0, False
            return 1.0, False

        # what the cameras see right now
        live = self._scene(now)
        # Moving obstacles: preds = where they are predicted to be, excl = where they have been lately.
        # They are handled by the prediction code below, so keep them out of the normal scene and the memory
        # (otherwise they leave a trail of fake standing obstacles behind them).
        if self.mv_on:
            self._track_update(now)
        preds, excl = self._movers(now) if self.mv_on else ([], [])
        if excl:
            live = [b for b in live if not self._inside_any(b, excl)]
        # on the way home the drone faces the other way, so the old memory is no use: live boxes only
        if self.state == "RTH":
            boxes = live
        else:
            boxes = self._remember(now, live, s_drone, cross, excl)
        if self.mv_on:
            boxes = boxes + self._hold_boxes(now)

        hw, zh = self.hw, self.zh
        # fx = distance to the nearest obstacle in the corridor ahead (half-width hw, only things near our height)
        fx = first_hit_x(boxes, 0.3, self.slow_dist, -hw, hw, zh)
        self.fx = fx
        # MOVING OBSTACLES. mv_scale = the fastest forward speed (0..1) with no predicted crash,
        # urgent = even standing still would get hit, so dodge right now.
        lat_v = self.evade_lat_speed if self.evading else self.lat_speed
        mv_scale, urgent, lane_free = 1.0, False, None
        if preds:
            # the drone is at y = 0 in this frame; the lane we are heading for is (cross - off_goal) to the left
            mv_scale, urgent = plan_vs_movers(preds, cross - self.off_goal, self.speed, lat_v, self.mv_hw, zh, self.mv_horizon, self.mv_brake)
            if urgent:
                self.evading = True
                lat_v = self.evade_lat_speed
                self._log_every(now, 2.0, "Moving obstacle will hit us even if we stop: dodging now.")
            elif mv_scale < 1.0:
                self._log_every(now, 2.0, f"Moving obstacle: slowing down to {mv_scale:.0%}.")
            # is it safe to slide to lane L (right +) and stay there?
            # Is lane L (right +) safe? Slide there (without moving forward), and unless we must dodge RIGHT NOW,
            # also check that we can then carry on forward along it.
            v_after = None if urgent else self.speed
            lane_free = lambda L: first_conflict(preds, cross - L, 0.0, lat_v, self.mv_hw, zh, self.mv_horizon,
                                                 v_after=v_after, slack=0.3, soft_slide=True) is None
            # standing still for a moving obstacle counts as blocked (so the give-up timer and lane search apply)
            if mv_scale == 0.0 or urgent:
                self.last_blocked_t = now
                if self.block_t0 is None:
                    self.block_t0 = now
                    self.get_logger().info("Moving obstacle ahead: stopping.")
        # BLOCKED / CLEAR: blocked = an obstacle at or inside stop_dist + 0.5 m. Once blocked, we only count as
        # clear again after clear_time seconds with nothing in the way (so flickering cannot make us creep forward).
        if fx <= self.stop_dist + 0.5:   # declared blocked just before the speed fades to zero (no endless creeping)
            self.last_blocked_t = now
            if self.block_t0 is None:
                self.block_t0 = now
                self.get_logger().info(f"Obstacle {fx:.1f} m ahead: stopping.")
        elif self.block_t0 is not None and now - self.last_blocked_t > self.clear_time:
            self.block_t0 = None
            self.get_logger().info("Path ahead is clear.")
        # blocked = the block timer is running, moving = a sideways move is under way,
        # side_ok = may we use the left / right camera
        blocked = self.block_t0 is not None
        moving = abs(self.off_goal - self.off_cmd) > 1e-3
        side_ok = self._side_ok(now)

        # Stuck for too long (give_up_time): give up. The first time -> recenter, then go home.
        # (The RECENTER check inside only applies if state is RECENTER, but this function is only called in FLY and RTH.)
        if blocked and now - self.block_t0 > self.give_up_time:
            if self.state == "RECENTER":
                self.get_logger().warn("Recovery also blocked: landing. " + self._why_no_lane(now))
                return 0.0, True
            self.get_logger().warn("Blocked too long with no way round: recentering before RTH. " + self._why_no_lane(now))
            return 0.0, "RECENTER"

        # Not sliding sideways right now, so we may decide on a new sideways move.
        no_lane = False
        if not moving:
            # CASE 1: blocked for pause_time -> look for a lane to go round the obstacle
            # (a thing that is still new gets mv_probe_time seconds to show whether it moves, before we detour around it)
            if blocked and (urgent or (now - self.block_t0 > self.pause_time and not (self.mv_on and self._probing(now)))):
                lanes = self.lanes
                # already going round something on one side: only allow lanes further out on that side
                # (or a small step back), or a big jump to the other side. This stops it wobbling back and forth.
                # (Not when a moving obstacle is about to hit us: then every lane is allowed.)
                if self.dodge_side and not urgent:
                    cur_abs = abs(cross)
                    side = self.dodge_side
                    lanes = [l for l in self.lanes
                             if l == 0.0
                             or (l * side > 0 and abs(l) > cur_abs - 0.5)        # further out (or small retreat)
                             or (l * side < 0 and abs(l) > cur_abs + self.lane_step)]  # jump over to the other side

                # pick the cheapest safe lane (None = no lane is safe)
                lane = choose_lane(boxes, cross, lanes, hw, zh, self.plan_dist, self.pass_margin, side_ok, self.left_bias, lane_free)
                # about to be hit and no lane is completely clear: take the lane that keeps us furthest away
                if lane is None and urgent and preds:
                    gaps = {L: min_gap(preds, cross - L, 0.0, lat_v, self.mv_hw, zh, self.mv_horizon) for L in lanes}
                    g_best = max(gaps.values())
                    if g_best > 0.1:
                        lane = choose_lane(boxes, cross, lanes, hw, zh, self.plan_dist, self.pass_margin, side_ok,
                                           self.left_bias, lambda L: gaps[L] >= g_best - 0.3)
                
                # found a lane different from where we are: make it the goal
                # (the sideways move itself happens further down)
                if lane is not None and abs(lane - self.off_cmd) > 1e-3:
                    if lane != 0.0:
                        # first dodge: remember how far along the route it started
                        if not self.dodge_side:
                            self.dodge_s = s_drone
                        self.dodge_side = 1.0 if lane > 0 else -1.0
                    self.off_goal = lane
                    self.get_logger().info(
                        f"Going around: moving to {lane:+.1f} m ({'right' if lane > self.off_cmd else 'left'}) of the line.")
                # no lane is available
                elif lane is None:
                    no_lane = urgent
                    # after waiting a bit longer, let it consider the other side too; otherwise keep waiting and log why
                    if self.dodge_side and now - self.block_t0 > self.pause_time + 6.0:
                        self.dodge_side = 0.0
                        self.get_logger().info("Nothing further out on that side: considering the other side too.")
                    else:
                        self._log_every(now, 5.0, "No clear lane yet. " + self._why_no_lane(now))
            # CASE 2: not blocked and we are off the line: go back to the line once we are far enough past
            # the obstacle (min_pass), or the carrot has already reached the end of the leg
            elif not blocked and abs(self.off_cmd) > 1e-3 and (s_drone - self.dodge_s >= self.min_pass or self.leg_done):
                # only go back if the move back AND the line ahead are both clear
                lane = choose_lane(boxes, cross, [0.0], hw, zh, self.plan_dist, self.pass_margin, side_ok, 0.0, lane_free)
                if lane is not None:
                    self.off_goal = 0.0
                    self.get_logger().info("Clear of the obstacle: returning to the line.")

        # a moving obstacle is now predicted to hit us in the lane we are sliding to: pick another lane and keep
        # sliding (stopping half way would leave us in its path). If there is no other lane, carry on to the same one.
        if preds and abs(self.off_goal - self.off_cmd) > 1e-3 and now - self.mv_replan_t >= self.mv_replan_gap and \
                first_conflict(preds, cross - self.off_goal, 0.0, lat_v, self.mv_hw, zh, self.mv_horizon,
                               v_after=v_after, soft_slide=True) is not None:
            # try lanes further out on the SAME side first (no wobbling across the obstacle's path),
            # and only if none of those is safe, try every lane
            out = [l for l in self.lanes if l * self.off_goal > 0 and abs(l) > abs(self.off_goal) + 1e-6]
            alt = choose_lane(boxes, cross, out, hw, zh, self.plan_dist, self.pass_margin, side_ok, self.left_bias, lane_free) if out else None
            if alt is None:
                alt = choose_lane(boxes, cross, self.lanes, hw, zh, self.plan_dist, self.pass_margin, side_ok, self.left_bias, lane_free)
            if alt is not None and abs(alt - self.off_goal) > 1e-3:
                self.off_goal = alt
                self.mv_replan_t = now
                self.get_logger().info(f"Moving obstacle: changing the sideways move to {alt:+.1f} m.")
        if abs(self.off_goal - self.off_cmd) > 1e-3:   # a sideways move is under way: check it every tick
            d = self.off_goal - self.off_cmd
            sgn = 1.0 if d > 0 else -1.0
            y_c = -sgn * min(abs(d), 2.0)
            # is anything in the way of the next ~2 m of sideways move (or just behind us)?
            hit = first_hit_x(boxes, -self.pass_margin, hw, min(0.0, y_c) - hw, max(0.0, y_c) + hw, zh)
            # something appeared, or that side's camera stopped: cancel the move
            if not side_ok(sgn) or hit < math.inf:
                self.off_goal = self.off_cmd
                self.get_logger().warn("Sideways move blocked: stopping and re-planning.")
            else:
                # otherwise slide sideways a little (lat_speed, per tick)
                self.off_cmd += sgn * min(abs(d), lat_v * dt)

        # the sideways move is finished and nothing is about to hit us: back to the calm sideways speed
        if not urgent and abs(self.off_goal - self.off_cmd) <= 1e-3:
            self.evading = False
        # back on the line and not blocked: forget which side we were dodging
        if abs(self.off_cmd) < 1e-3 and abs(self.off_goal) < 1e-3 and not blocked:
            self.dodge_side = 0.0
        # scale = forward speed from 0 (stop) to 1 (full). It fades from full at slow_dist down to 0 at
        # stop_dist, and is 0 whenever we are blocked.
        scale = 1.0
        if fx < self.slow_dist:
            scale = max(0.0, min(1.0, (fx - self.stop_dist) / (self.slow_dist - self.stop_dist)))
        if blocked:
            scale = 0.0
        # a moving obstacle may ask for a lower speed than the standing obstacles do
        scale = min(scale, mv_scale)
        # about to be hit, hovering would not help and there is no lane to dodge into: back away along the route
        self.accel_now = self.accel
        if no_lane and urgent and self.mv_retreat > 0 and abs(self.off_goal - self.off_cmd) <= 1e-3:
            # v7: already level with the thing (inside its slab along the route): backing away would keep us in its
            # path, so run FORWARD out of it (if nothing standing is close ahead). Otherwise back away as before.
            inside = any(abs(cy) < hy + pad0 + self.mv_hw + 0.5 and cx - hx - pad0 - self.mv_hw < 0.3 and cx + hx + pad0 + self.mv_hw > 0.0
                         for cx, cy, hx, hy, z0, z1, vx, vy, pad0, growth in preds if not (z1 < -zh or z0 > zh))
            if inside and fx > 3.0:
                scale = 1.0
                self.accel_now = max(self.accel, 2.0)
                self._log_every(now, 2.0, "Moving obstacle is level with us: leaving its path forward.")
            else:
                scale = -min(1.0, self.mv_retreat / self.speed)
                self._log_every(now, 2.0, "No lane to dodge into: backing away from the moving obstacle.")
        # v7 STANDOFF: never hover right next to something. Too close to a standing obstacle, a moving one or a sway
        # stripe straight ahead -> back away along the route (if the camera behind is running and the way back is clear).
        too_close = fx < self.standoff
        for cx, cy, hx, hy, z0, z1, vx, vy, pad0, growth in preds:
            if z1 < -zh or z0 > zh or cx < -0.2 or abs(cy) >= hy + pad0 + self.mv_hw:
                continue
            if cx - hx - pad0 - self.mv_hw < self.mv_standoff:
                too_close = True
        self.lead_now = 0.0 if (too_close or self.fx < self.stop_dist or preds) else self.brake_lead
        if too_close and self.mv_retreat > 0 and scale >= -1e-9 and (self._alive("back", now) or self.allow_blind) \
                and first_hit_x(boxes, -3.0, -0.2, -hw, hw, zh) == math.inf:
            scale = -min(1.0, self.mv_retreat / self.speed)
            self._log_every(now, 2.0, "Too close to an obstacle ahead: backing away to keep a safe distance.")
        return scale, False

    def _avoid_rth(self, now, dt, cross, s_drone):
        """Same as _avoid, but on give-up, return (0, True) → land. No second RECENTER."""
        scale, result = self._avoid(now, dt, cross, s_drone)
        if result == "RECENTER":
            # In RTH, don't RECENTER again, land.
            self.get_logger().warn("RTH: blocked again with no way around — landing in place.")
            return 0.0, True
        return scale, result

    # Start the route: the first leg begins at the drone's current position.
    def _begin_flight(self, now):
        self.seg = 0
        self.seg_start = self.pos.copy()
        self.carrot_s = 0.0
        self.fly_t0 = now
        self.mem = []
        self.leg_done = False
        self._set_state("FLY")

    # Start RECENTER: slide back onto the route line before heading home.
    # cross = how far off the line we are, r = the sideways direction.
    def _begin_recenter(self, now, cross, r):
        # the decision to give up on the destination and go home
        self.returning = True
        self.giveup_reason = f"blocked for over {self.give_up_time:.0f} s with no way round"
        # already on the line: go straight to RTH
        if abs(cross) < 0.1:
            self.get_logger().info("RECENTER: already on line, backtracking before RTH.")
            self._begin_return_home(now)
            return
        # remember how far we still have to slide, in which direction, and the point we slide from
        self.recenter_offset = cross
        self.recenter_r = r.copy()
        self.recenter_carrot = self.pos.copy()
        self.get_logger().info(
            f"RECENTER: sliding from {cross:+.1f} m to 0 m before backtrack.")
        # reset the 'blocked' timer
        self.block_t0 = None
        self.last_blocked_t = -1e9
        self._set_state("RECENTER")
        
    # Start RTH (return to home): reverse the route and fly it back.
    def _begin_return_home(self, now):
        # we never saved a home point, so there is nowhere to go: land here
        if self.home_pos is None:
            self.get_logger().warn("No home position captured: landing in place.")
            self.last_cmd_t = 0.0
            self._set_state("LAND")
            return
        self.get_logger().info(
            f"Returning to spawn at NED {np.round(self.home_pos, 1).tolist()}")
        # Build reverse route: from current position → waypoints backwards → home.
        # The current leg is self.wps[self.seg] with start self.seg_start.
        route = []
        for i in range(self.seg - 1, -1, -1):
            route.append(self.wps[i].copy())
        route.append(self.home_pos.copy())
        self.wps = np.array(route)
        # reset the flight bookkeeping for the new (reversed) route
        self.seg = 0
        self.seg_start = self.pos.copy()
        self.carrot_s = 0.0
        self.v = 0.0
        self.off_cmd = self.off_goal = 0.0
        self.dodge_side = 0.0
        self.mem = []
        self.leg_done = False
        # throw away old camera boxes: they were measured while facing the other way
        self.hist = {c: [] for c in CAMS}
        self.block_t0 = None
        self.last_blocked_t = -1e9
        self._set_state("RTH")

    # Last waypoint reached: land, or hold still (depends on land_at_end).
    def _finish_route(self):
        self.get_logger().info("Route complete.")
        self.outcome = "arrived"
        self._print_summary()
        if self.land_at_end:
            self.last_cmd_t = 0.0
            self._set_state("LAND")
        else:
            self.hold_pos = self.wps[-1].copy()
            self._set_state("HOLD")

    # Print the cross-track error stats (how far off the line the drone was): mean, RMSE and max.
    def _print_summary(self):
        if self.xte_n:
            self.get_logger().info(
                f"Cross-track error over {self.xte_n} samples: mean {self.xte_sum / self.xte_n:.2f} m, "
                f"RMSE {math.sqrt(self.xte_sq / self.xte_n):.2f} m, max {self.xte_max:.2f} m")

    # ------------------------------------------------------------------ dashboard status
    # Is the current leg the straight-up takeoff leg?
    def _leg_is_vertical(self):
        if self.wps is None or self.seg_start is None or self.seg >= len(self.wps):
            return False
        v = self.wps[self.seg] - self.seg_start
        return math.hypot(v[0], v[1]) <= 1.0

    # What is the drone doing, in a form the dashboard can show (a small JSON-friendly dict)?
    def _mission_status(self, now):
        s = self.state
        blocked = self.block_t0 is not None
        sliding = abs(self.off_goal - self.off_cmd) > 1e-3
        off_line = abs(self.off_cmd) > 0.05
        slowing = math.isfinite(self.fx) and self.fx < self.slow_dist
        avoiding = s in ("FLY", "RTH") and (blocked or sliding or off_line or self.evading or slowing)
        why = ""
        if avoiding:
            ahead = f"{self.fx:.1f} m ahead" if math.isfinite(self.fx) else "ahead"
            if self.off_goal != 0.0:
                why = f"going around: {abs(self.off_goal):.1f} m {'right' if self.off_goal > 0 else 'left'} of the line"
            elif sliding or off_line:
                why = "obstacle passed: returning to the line"
            elif blocked:
                why = f"obstacle {ahead}: stopped, looking for a way round"
            elif slowing:
                why = f"obstacle {ahead}: slowing down"
            else:
                why = "dodging a moving obstacle"
        if s in ("WAIT", "STREAM", "ENGAGE"):
            phase, detail = "preparing", "waiting for PX4, then arming"
        elif s == "FLY":
            if self._leg_is_vertical():
                phase, detail = "takeoff", "climbing to the route altitude"
            else:
                phase = "avoiding" if avoiding else "following"
                detail = why if avoiding else "on the planned route"
        elif s == "RECENTER":
            phase, detail = "recentering", "no way round: moving back onto the line before heading home"
        elif s == "RTH":
            phase = "returning"
            if now - getattr(self, "rth_t0", now) < 2.0:
                detail = "turning around"
            else:
                detail = why if avoiding else "flying back to the start"
        else:                                           # HOLD, LAND, DONE: say how the mission ended
            phase = self.outcome or "aborted"
            ended = {"arrived": "final destination reached",
                     "returned": "final destination unapproachable: flew back to the start",
                     "aborted": "blocked with no way round: landing in place"}[phase]
            detail = ended + {"LAND": " (landing)", "DONE": " (landed)", "HOLD": " (holding)"}.get(s, "")
        return {"type": "mission", "phase": phase, "avoiding": bool(avoiding), "detail": detail, "state": s,
                "wp": [min(self.seg + 1, len(self.wps)), len(self.wps)] if self.wps is not None else None,
                "obstacle_m": round(float(self.fx), 1) if math.isfinite(self.fx) else None,
                "offset_m": round(float(self.off_cmd), 1), "reason": self.giveup_reason or None}

    # The flown path (as lat/lon so the dashboard can draw it) with a short summary.
    def _build_complete(self, now):
        if self.pos is None:
            return None
        leg = 1 if self.returning else 0
        pts = list(self.path) + [(float(self.pos[0]), float(self.pos[1]), float(self.pos[2]), leg)]
        dist = sum(math.dist(a[:3], b[:3]) for a, b in zip(pts, pts[1:]))
        msg = {"type": "mission_complete", "outcome": self.outcome or "aborted", "reason": self.giveup_reason or None,
               "distance_m": round(dist, 1), "duration_s": round(now - self.fly_t0, 1) if self.fly_t0 else None,
               "path": None}
        if self.xte_n:
            msg["xte_mean_m"] = round(self.xte_sum / self.xte_n, 2)
            msg["xte_max_m"] = round(self.xte_max, 2)
        if self.ref is not None:                       # lat/lon needs the origin PX4 told us about
            msg["path"] = []
            for n, e, d, lg in pts:
                lat, lon, up = ned_to_gps(n, e, d, *self.ref)
                msg["path"].append([round(lat, 7), round(lon, 7), round(up, 1), lg])
        return msg

    def _save_flown_route(self, msg):
        folder = str(self.cfg["flown_route_dir"])
        if not folder:
            return
        try:
            folder = os.path.expanduser(folder)
            os.makedirs(folder, exist_ok=True)
            name = os.path.join(folder, datetime.now().strftime("flown_route_%Y%m%d_%H%M%S.json"))
            with open(name, "w") as fh:
                json.dump(msg, fh)
            self.get_logger().info(f"Flown route saved to {name}")
        except OSError as e:
            self.get_logger().warn(f"Could not save the flown route: {e}")

    # Runs after every tick of the main loop: remember the path, publish the status, publish the finished route.
    def _mission_tick(self, now):
        # 1) remember where the drone has been (a point every 0.5 m)
        if self.pos is not None and self.state in ("FLY", "RECENTER", "RTH", "LAND"):
            if not self.path or math.dist(self.pos, self.path[-1][:3]) >= 0.5:
                self.path.append((float(self.pos[0]), float(self.pos[1]), float(self.pos[2]), 1 if self.returning else 0))
                if len(self.path) > 3000:
                    self.path = self.path[::2]
        # 2) status, twice a second
        if now - self.mission_t >= 0.5:
            self.mission_t = now
            self.mission_pub.publish(String(data=json.dumps(self._mission_status(now))))
        # 3) the mission has ended: build the finished route once, then keep publishing it (the dashboard keeps the latest)
        if self.state == "LAND" and self.land_t0 is None:
            self.land_t0 = now
        ended = self.state in ("HOLD", "DONE") or (self.state == "LAND" and now - self.land_t0 > 20.0)
        if ended and self.complete_msg is None:
            msg = self._build_complete(now)
            if msg:
                self.complete_msg = json.dumps(msg)
                self.get_logger().info(f"Mission ended ({msg['outcome']}): {msg['distance_m']} m flown.")
                self._save_flown_route(msg)
        if self.complete_msg and now - self.complete_t >= 5.0:
            self.complete_t = now
            self.mission_pub.publish(String(data=self.complete_msg))

    # ------------------------------------------------------------------ main loop
    # Runs rate_hz times a second. The dashboard code can never be allowed to stop the flight code.
    def on_timer(self):
        self._tick()
        try:
            self._mission_tick(self._now())
        except Exception as e:
            self._log_every(self._now(), 5.0, f"Dashboard status error: {e!r}")

    # The main loop. Runs rate_hz times a second; does the job of the current state, then returns.
    def _tick(self):
        now = self._now()
        # dt = seconds since the last tick (kept between 0 and 0.2 so a hiccup cannot cause a big jump)
        dt = 0.0 if self.t_last is None else min(max(now - self.t_last, 0.0), 0.2)
        self.t_last = now

        # WAIT: wait until PX4 has told us everything we need (position, status, GPS origin)
        if self.state == "WAIT":
            self.publish_heartbeat()
            # GPS waypoints: convert them to local metres once we know where (0, 0) is
            if self.use_gps and self.wps is None and self.ref is not None:
                self.wps = np.array([gps_to_ned(la, lo, up, *self.ref) for la, lo, up in self.wp_raw])
                self.get_logger().info(f"GPS origin {self.ref} -> waypoints (NED): {np.round(self.wps, 1).tolist()}")
            # got everything: remember this spot to hold at, and move on to STREAM
            if self.pos is not None and self.pos_valid and self.have_status and self.wps is not None:
                self.hold_pos = self.pos.copy()
                self.hold_yaw = self.heading
                self._set_state("STREAM")
            else:
                self._log_every(now, 5.0, "Waiting for PX4 position, status and GPS origin...")
            return

        # STREAM then ENGAGE: PX4 refuses offboard mode unless it has already been receiving setpoints.
        # So first STREAM about 1 second of 'stay here' setpoints, then ENGAGE = ask for offboard mode and arm.
        if self.state in ("STREAM", "ENGAGE"):
            self.publish_heartbeat()
            self.publish_setpoint(self.hold_pos, self.hold_yaw)
            if self.state == "STREAM":
                self.stream_count += 1
                if self.stream_count >= int(self.rate_hz):
                    self.engage_t0 = now
                    self._set_state("ENGAGE")
                return
            # offboard + armed: start the route
            if self.armed and self.offboard:
                self._begin_flight(now)
                return
            # not yet: ask again once a second (mode 6 = offboard)
            if now - self.last_cmd_t >= 1.0:
                self.last_cmd_t = now
                self.send_command(CMD_MODE, 1.0, 6.0)
                self.send_command(CMD_ARM, 1.0)
                self.get_logger().info("Requesting offboard mode + arm...")
            # still not working after 10 s: tell the user to look for the reason
            if now - self.engage_t0 > 10.0:
                self._log_every(now, 5.0, "Still not armed/offboard. Check QGroundControl for the reason.")
            return

        # FLY: follow the route and avoid obstacles
        if self.state == "FLY":
            # The current leg goes from seg_start to target. vec / length = the leg, d3 = its direction (length 1).
            target = self.wps[self.seg]
            vec = target - self.seg_start
            length = float(np.linalg.norm(vec))
            d3 = vec / length if length > 1e-6 else np.zeros(3)
            # hvec / hlen = the same leg, but only along the ground (height ignored)
            hvec = np.array([vec[0], vec[1], 0.0])
            hlen = float(np.linalg.norm(hvec))
            # s_drone = how far along the leg the drone itself is
            s_drone = float((self.pos - self.seg_start) @ d3) if length > 1e-6 else 0.0
            if hlen > 1.0:   # a proper horizontal leg: obstacle avoidance is active
                u = hvec / hlen
                r = np.array([-u[1], u[0], 0.0])  # sideways direction, right of travel
                # face along the route
                self.hold_yaw = math.atan2(u[1], u[0])
                # cross = how far the drone is to the right of the line (negative = left)
                cross = float((self.pos - self.seg_start) @ r)
                self.leg_done = length > 1e-6 and self.carrot_s >= length - 1e-6
                # ask the avoidance code: scale = how fast to go (0..1), land = True or 'RECENTER' if we must give up
                scale, land = self._avoid(now, dt, cross, s_drone)
                # stuck: slide back to the line, then go home
                if land == "RECENTER":
                    self._begin_recenter(now, cross, r)
                    return
                
                # stuck again: land
                if land:
                    self.last_cmd_t = 0.0
                    self._set_state("LAND")
                    return
                
            else:            # the straight-up takeoff leg
                r = np.zeros(3)
                cross = 0.0
                self.off_cmd = self.off_goal = 0.0
                self.fx = math.inf
                scale = 1.0

            s_drone = float((self.pos - self.seg_start) @ d3) if length > 1e-6 else 0.0
            # SPEED: speed up at 'accel' (gently) or slow down at 'decel' (quickly), aiming for speed * scale
            dv = self.speed * scale - self.v
            self.v += max(-self.decel * dt, min(self.accel_now * dt, dv))
            # MOVE THE CARROT forward, but only if it is not already more than max_lead ahead of the drone
            if length > 1e-6 and (self.carrot_s - s_drone) < self.max_lead:
                self.carrot_s = min(self.carrot_s + self.v * dt, length)
                if self.v < 0:      # backing away: the carrot must not run off behind the drone or before the start of the leg
                    self.carrot_s = max(self.carrot_s, 0.0, s_drone - self.max_lead)
            if scale <= 0.0 and length > 1e-6:    # blocked: stop the carrot at the drone
                self.carrot_s = min(self.carrot_s, max(s_drone, 0.0) + self.lead_now)
            # carrot position: along the line, then shifted sideways by the lane offset
            base = self.seg_start + d3 * self.carrot_s if length > 1e-6 else target.copy()
            carrot = base + r * self.off_cmd

            # send it to PX4
            self.publish_heartbeat()
            self.publish_setpoint(carrot, self.hold_yaw)

            # CROSS-TRACK ERROR = distance from the original route line: publish it and add it to the running stats
            xte = abs(cross) if hlen > 1.0 else point_segment_distance(self.pos, self.seg_start, target)
            self.xte_pub.publish(Float32(data=float(xte)))
            self.xte_sum += xte
            self.xte_sq += xte * xte
            self.xte_max = max(self.xte_max, xte)
            self.xte_n += 1
            # save a row to the CSV log if it is turned on
            if self.csv:
                self.csv.writerow([f"{now - self.fly_t0:.2f}", *[f"{v:.3f}" for v in self.pos],
                                   *[f"{v:.3f}" for v in carrot], f"{cross:.3f}", f"{self.off_cmd:.2f}",
                                   f"{self.fx:.2f}", self.seg])
            # warn once if PX4 has left offboard mode
            if not self.offboard and not self.warned_offboard:
                self.warned_offboard = True
                self.get_logger().warn("PX4 left offboard mode (manual takeover or failsafe?).")

            # distance to the target; print progress once a second
            dist = float(np.linalg.norm(self.pos - target))
            self._log_every(now, 1.0, f"FLY target {self.seg + 1}/{len(self.wps)} dist {dist:.1f} m, "
                                      f"offset {cross:+.1f} m, ahead-clear {self.fx:.1f} m, v {self.v:.2f}")
            # end of the leg: the carrot has finished AND the drone is within accept_radius of the target
            if self.carrot_s >= length - 1e-6 and dist < self.accept_radius:
                self.get_logger().info(f"Reached waypoint {self.seg + 1}/{len(self.wps)}")
                # the first waypoint reached is saved as 'home' (where RTH will return to)
                if self.home_pos is None:
                    self.home_pos = self.pos.copy()
                    self.home_yaw = self.hold_yaw
                    self.get_logger().info(
                        f"Home position captured at NED {np.round(self.home_pos, 1).tolist()}")
                # start the next leg from here, and reset the per-leg bookkeeping
                self.seg_start = target.copy()
                self.seg += 1
                self.carrot_s = 0.0
                self.off_cmd = self.off_goal = 0.0
                self.dodge_side = 0.0
                self.leg_done = False
                self.mem = []    # the next leg has a different frame
                # that was the last leg
                if self.seg >= len(self.wps):
                    self._finish_route()
            return

        # RECENTER: slide sideways back onto the route line at lat_speed, then start RTH.
        if self.state == "RECENTER":
            # close enough to the line (within 5 cm): start RTH
            if abs(self.recenter_offset) < 0.05:
                self.get_logger().info("RECENTER: done, entering RTH.")
                self.recenter_offset = 0.0
                self._begin_return_home(now)
                return
            # move a small step towards the line this tick (never past it)
            step = min(abs(self.recenter_offset), self.lat_speed * dt)
            step_signed = math.copysign(step, self.recenter_offset)
            self.recenter_carrot = self.recenter_carrot - self.recenter_r * step_signed
            self.recenter_offset -= step_signed
            self.publish_heartbeat()
            self.publish_setpoint(self.recenter_carrot, self.hold_yaw)
            self._log_every(now, 1.0, f"RECENTER: off {self.recenter_offset:+.1f} m")
            return
        
        # RTH: fly the reversed route back home. It works like FLY (same dodging), but does not
        # collect cross-track stats, and landing replaces a second RECENTER.
        if self.state == "RTH":
            # Pre-turn hold: give the cameras 2 s to refill with post-turn boxes.
            if not hasattr(self, "rth_t0"):
                self.rth_t0 = now
                
            if now - self.rth_t0 < 2.0:
                self.publish_heartbeat()
                self.publish_setpoint(self.pos.copy(), self.hold_yaw)
                self._log_every(now, 1.0, "RTH: turning around...")
                return
            
            # From here to the end of the RTH block, the code is the same as in FLY above:
            # work out the leg, ask the avoidance code for a speed, move the carrot, send it to PX4.
            target = self.wps[self.seg]
            vec = target - self.seg_start
            length = float(np.linalg.norm(vec))
            d3 = vec / length if length > 1e-6 else np.zeros(3)
            hvec = np.array([vec[0], vec[1], 0.0])
            hlen = float(np.linalg.norm(hvec))
            s_drone = float((self.pos - self.seg_start) @ d3) if length > 1e-6 else 0.0
            if hlen > 1.0:
                u = hvec / hlen
                r = np.array([-u[1], u[0], 0.0])
                self.hold_yaw = math.atan2(u[1], u[0])
                cross = float((self.pos - self.seg_start) @ r)
                self.leg_done = length > 1e-6 and self.carrot_s >= length - 1e-6
                # In RTH we still dodge, but if we give up a second time, land.
                scale, land = self._avoid_rth(now, dt, cross, s_drone)
                if land:
                    self.last_cmd_t = 0.0
                    self._set_state("LAND")
                    return
            else:
                r = np.zeros(3)
                cross = 0.0
                self.off_cmd = self.off_goal = 0.0
                self.fx = math.inf
                scale = 1.0

            s_drone = float((self.pos - self.seg_start) @ d3) if length > 1e-6 else 0.0
            dv = self.speed * scale - self.v
            self.v += max(-self.decel * dt, min(self.accel_now * dt, dv))
            if length > 1e-6 and (self.carrot_s - s_drone) < self.max_lead:
                self.carrot_s = min(self.carrot_s + self.v * dt, length)
                if self.v < 0:      # backing away: the carrot must not run off behind the drone or before the start of the leg
                    self.carrot_s = max(self.carrot_s, 0.0, s_drone - self.max_lead)
            if scale <= 0.0 and length > 1e-6:
                self.carrot_s = min(self.carrot_s, max(s_drone, 0.0) + self.lead_now)
            base = self.seg_start + d3 * self.carrot_s if length > 1e-6 else target.copy()
            carrot = base + r * self.off_cmd

            self.publish_heartbeat()
            self.publish_setpoint(carrot, self.hold_yaw)

            if not self.offboard and not self.warned_offboard:
                self.warned_offboard = True
                self.get_logger().warn("PX4 left offboard mode.")

            dist = float(np.linalg.norm(self.pos - target))
            self._log_every(now, 1.0, f"RTH dist {dist:.1f} m, offset {cross:+.1f} m, "
                                      f"ahead-clear {self.fx:.1f} m, v {self.v:.2f}")
            # end of the leg: start the next one; after the last one, land at home
            if self.carrot_s >= length - 1e-6 and dist < self.accept_radius:
                self.get_logger().info(f"RTH: reached waypoint {self.seg + 1}/{len(self.wps)}")
                self.seg_start = target.copy()
                self.seg += 1
                self.carrot_s = 0.0
                self.off_cmd = self.off_goal = 0.0
                self.dodge_side = 0.0
                self.leg_done = False
                self.mem = []
                if self.seg >= len(self.wps):
                    self.get_logger().info("RTH: reached spawn, landing.")
                    self.outcome = "returned"
                    self.hold_pos = self.home_pos.copy()
                    self.hold_yaw = self.home_yaw
                    self.last_cmd_t = 0.0
                    self._set_state("LAND")
            return

        # HOLD: stay put at hold_pos (the route is finished and land_at_end is off)
        if self.state == "HOLD":
            self.publish_heartbeat()
            self.publish_setpoint(self.hold_pos, self.hold_yaw)
            return

        # LAND: send the land command once a second until the drone has disarmed, then DONE
        if self.state == "LAND":
            if now - self.last_cmd_t >= 1.0:
                self.last_cmd_t = now
                self.send_command(CMD_LAND)
                self.get_logger().info("Landing...")
            if not self.armed:
                self._set_state("DONE")
            return

    # Close the CSV file when shutting down.
    def close(self):
        if self.csv_file:
            self.csv_file.close()

# Start ROS, run the node until Ctrl+C or otherwise killed, then shut down cleanly.
def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
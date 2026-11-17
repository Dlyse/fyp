#!/usr/bin/env python3
"""Publish the ROUTE list from waypoint_follower.py to the dashboard.

Reads the file as text (it never runs it), so px4_msgs is not needed, and it
re-reads the file whenever you save a change.

    source /opt/ros/jazzy/setup.bash
    python3 publish_route.py                          # reads ~/waypoint_follower.py
    python3 publish_route.py --file /path/to/waypoint_follower.py

ROUTE in that file must look like:  ROUTE = [(lat, lon, alt_m), (lat, lon, alt_m), ...]
The drone client (ROS2=1 drone_client.py) forwards what is published on /dashboard/route.
"""
import argparse
import ast
import json
import os


def read_route(path):
    """Return [[lat, lon, alt], ...] from the top-level ROUTE = [...] in a python file."""
    with open(os.path.expanduser(path)) as f:
        tree = ast.parse(f.read())
    for node in tree.body:                      # top-level assignments only
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        if any(n.lower() == "route" for n in names):
            pts = ast.literal_eval(node.value)
            return [[float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0] for p in pts]
    raise ValueError("no top-level ROUTE = [...] found")


def make_message(points):
    return json.dumps({"frame": "latlon", "points": points})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="~/waypoint_follower.py", help="python file containing ROUTE")
    ap.add_argument("--topic", default="/dashboard/route")
    args = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class RoutePublisher(Node):
        def __init__(self):
            super().__init__("route_publisher")
            self.path = os.path.expanduser(args.file)
            self.pub = self.create_publisher(String, args.topic, 10)
            self.points, self.seen = [], "unset"
            self.create_timer(1.0, self.tick)

        def tick(self):
            try:
                mtime = os.path.getmtime(self.path)
            except OSError:
                mtime = None
            if mtime != self.seen:                       # file appeared, was saved again, or vanished
                self.seen = mtime
                if mtime is None:
                    self.get_logger().warn(f"{self.path} not found")
                else:
                    try:
                        self.points = read_route(self.path)
                        self.get_logger().info(f"Loaded {len(self.points)} waypoints from {self.path}")
                    except Exception as e:                # warn once, keep the last good route
                        self.get_logger().warn(f"Could not read ROUTE from {self.path}: {e}")
            if self.points:
                self.pub.publish(String(data=make_message(self.points)))   # repeated so late listeners get it

    rclpy.init()
    node = RoutePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

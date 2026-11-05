from pymavlink import mavutil

m = mavutil.mavlink_connection('udpin:0.0.0.0:14540')
m.wait_heartbeat()
print(f"Connected: system {m.target_system}, component {m.target_component}")

# Ask PX4 for position at 5 Hz (interval in microseconds)
m.mav.command_long_send(
    m.target_system, m.target_component,
    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
    mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 200000,
    0, 0, 0, 0, 0)

while True:
    msg = m.recv_match(type=['GLOBAL_POSITION_INT', 'GPS_RAW_INT'], blocking=True)
    if msg.get_type() == 'GLOBAL_POSITION_INT':
        lat, lon = msg.lat / 1e7, msg.lon / 1e7   # degrees
        alt_msl = msg.alt / 1000                  # metres above MSL
        alt_rel = msg.relative_alt / 1000         # metres above home
        vx, vy, vz = msg.vx / 100, msg.vy / 100, msg.vz / 100   # m/s
        hdg = msg.hdg / 100                       # degrees
        print(lat, lon, alt_msl, alt_rel, hdg)
    else:
        print("GPS fix type:", msg.fix_type, "sats:", msg.satellites_visible)

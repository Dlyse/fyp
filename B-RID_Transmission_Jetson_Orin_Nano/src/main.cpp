#include <Arduino.h>
#include <TinyGPSPlus.h>

#include "id_open.h"

static ID_OpenDrone          squitter;

static struct UTM_parameters utm_parameters;
static struct UTM_data       utm_data;

static TinyGPSPlus gps;
static HardwareSerial gpsSerial(2);
static const int GPS_RX_PIN = 17; // GPS TXD connects here
static const int GPS_TX_PIN = 16; // GPS RXD connects here

//float i = 0.0;

void setup() {

  Serial.begin(115200);

  gpsSerial.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);

  memset(&utm_parameters,0,sizeof(utm_parameters));

  strcpy(utm_parameters.UAS_operator,"UAE-SSRC-1234567");

  strcpy(utm_parameters.UAV_id, "TEST-DRONE-0001");
  utm_parameters.UA_type = ODID_UATYPE_HELICOPTER_OR_MULTIROTOR; //ODID_UATYPE_AEROPLANE for planes
  utm_parameters.ID_type = ODID_IDTYPE_SERIAL_NUMBER;

  utm_parameters.region      = 1;
  utm_parameters.EU_category = 2;

  squitter.init(&utm_parameters);
  
  memset(&utm_data,0,sizeof(utm_data));

  utm_data.base_latitude  = 0.0;
  utm_data.base_longitude = 0.0;//
  utm_data.base_alt_m     = 0.0;

  utm_data.latitude_d  = 0.0;//1.3 or i
  utm_data.longitude_d = 0.0;//103.717382
  utm_data.alt_msl_m   = 0.0; //mean sea lvl

  utm_data.alt_agl_m = 0.0; //heigh above take off point

  utm_data.satellites = 0;
  utm_data.base_valid = 0;

  return;
}

void loop() {
  while (gpsSerial.available() > 0) {
    char c = gpsSerial.read();
    Serial.write(c);
    gps.encode(c);
  }
 /*
  static unsigned long lastDebug = 0;
if (millis() - lastDebug >= 1000) {
  lastDebug = millis();

  Serial.print("Characters received: ");
  Serial.println(gps.charsProcessed());

  Serial.print("Location valid: ");
  Serial.println(gps.location.isValid());

  Serial.print("Location age: ");
  Serial.println(gps.location.age());

  Serial.print("Satellites valid: ");
  Serial.println(gps.satellites.isValid());

  Serial.print("Satellites: ");
  Serial.println(gps.satellites.value());

  Serial.println("----------------");
}
*/


  if (gps.location.isValid() && gps.location.age() < 2000) {

    utm_data.latitude_d = gps.location.lat();
    utm_data.longitude_d = gps.location.lng();

    if (gps.satellites.isValid()) {
      utm_data.satellites = gps.satellites.value();
    }

    if (gps.altitude.isValid()) {
      utm_data.alt_msl_m = gps.altitude.meters();
    }

    if (gps.speed.isValid()) {
      utm_data.speed_kn = round(gps.speed.knots());
    }

    if (gps.course.isValid()) {
      utm_data.heading = round(gps.course.deg());
    }

    if (gps.time.isValid()) {
      utm_data.minutes = gps.time.minute();
      utm_data.seconds = gps.time.second();
      utm_data.csecs = gps.time.centisecond();
    }

    // Save the first valid GPS position as the take-off location
    if (!utm_data.base_valid) {
      utm_data.base_latitude = utm_data.latitude_d;
      utm_data.base_longitude = utm_data.longitude_d;
      utm_data.base_alt_m = utm_data.alt_msl_m;
      utm_data.base_valid = 1;
    }

    // Height above take-off point
    utm_data.alt_agl_m = utm_data.alt_msl_m - utm_data.base_alt_m;
  // Serial.print("Satellites: ");
  // Serial.println(utm_data.satellites);

  // Serial.print("Latitude: ");
  // Serial.println(utm_data.latitude_d, 6);

  // Serial.print("Longitude: ");
  // Serial.println(utm_data.longitude_d, 6);

  // Serial.print("Altitude: ");
  // Serial.println(utm_data.alt_msl_m, 6);

  } else {
    // Prevent an invalid location from being transmitted
    utm_data.satellites = 0;//change to
  }

  squitter.transmit(&utm_data);
  // i+=0.0001;
  // utm_data.latitude_d = i;
}
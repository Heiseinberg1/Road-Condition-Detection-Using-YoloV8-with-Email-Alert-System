import cv2
import time
import csv
import os
import threading
from datetime import datetime
from picamera2 import Picamera2
import serial
import pynmea2
import smtplib
from email.message import EmailMessage

from detector import YOLOv8Detector

# ================= CONFIG =================
MODEL_PATH     = "/home/raspberrypi5/project/model/best.onnx"
IMAGE_FOLDER   = "/home/raspberrypi5/project/images/detected"
LOG_FILE       = "/home/raspberrypi5/project/logs/pothole_log.csv"

EMAIL            = "bibekkhanal122@gmail.com"
DEPARTMENT_EMAIL = "bibekkhanal.078@kathford.edu.np"
PASSWORD         = "fjdvdykbugorstxb"

ALERT_INTERVAL        = 30
CONFIDENCE_THRESHOLD  = 0.5


os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# ─── CSV header (write once if new file) ──────────────────
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(["timestamp", "latitude", "longitude", "image_path"])

# ================= INIT =================
detector = YOLOv8Detector(MODEL_PATH, CONFIDENCE_THRESHOLD)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

gps = serial.Serial("/dev/ttyAMA0", 9600, timeout=1)

# ================= BACKGROUND GPS READER =================
latest_gps = {"lat": "N/A", "lon": "N/A"}
gps_lock   = threading.Lock()

def gps_reader():
    while True:
        try:
            line = gps.readline().decode("ascii", errors="replace")
            if "$GPGGA" in line:
                msg = pynmea2.parse(line)
                if msg.latitude and msg.longitude:
                    with gps_lock:
                        latest_gps["lat"] = msg.latitude
                        latest_gps["lon"] = msg.longitude
        except Exception as e:
            print(f"GPS error: {e}")
            continue

threading.Thread(target=gps_reader, daemon=True).start()

# ================= FUNCTIONS =================

def send_email(image_path, lat, lon):
    try:
        msg = EmailMessage()
        msg['Subject'] = "Pothole Detection Alert"
        msg['From']    = EMAIL
        msg['To']      = DEPARTMENT_EMAIL

        maps_link = f"https://www.google.com/maps?q={lat},{lon}"
        msg.set_content(f"""Pothole Detected

Location : {lat}, {lon}
Maps     : {maps_link}
Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
        with open(image_path, 'rb') as f:
            msg.add_attachment(f.read(), maintype='image',
                               subtype='jpeg', filename="pothole.jpg")

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL, PASSWORD)
            smtp.send_message(msg)

        print("Email sent")

    except Exception as e:
        print(f"Email error: {e}")


def save_log(lat, lon, image_path):
    try:
        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([datetime.now(), lat, lon, image_path])
    except Exception as e:
        print(f"Log error: {e}")


def process_alert(frame, lock):
    try:
        # Grab latest GPS reading (non-blocking)
        with gps_lock:
            lat = latest_gps["lat"]
            lon = latest_gps["lon"]

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(IMAGE_FOLDER, f"pothole_{timestamp}.jpg")

        cv2.imwrite(image_path, frame)
        save_log(lat, lon, image_path)

        threading.Thread(
            target=send_email,
            args=(image_path, lat, lon),
            daemon=True
        ).start()

        print(f"Pothole logged — {lat}, {lon} — {image_path}")

    except Exception as e:
        print(f"Alert processing error: {e}")

    finally:
        lock.release()   # always release so next alert can fire


# ================= MAIN LOOP =================
last_alert  = 0
alert_lock  = threading.Lock()

print("System started — press Q to quit")

try:
    while True:
        frame = picam2.capture_array()

        # Convert RGBA → BGR (Pi camera returns 4-channel)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        pothole = detector.detect(frame)

        now = time.time()
        if pothole and (now - last_alert > ALERT_INTERVAL):
            if alert_lock.acquire(blocking=False):
                last_alert = now
                threading.Thread(
                    target=process_alert,
                    args=(frame.copy(), alert_lock),
                    daemon=True
                ).start()

        cv2.imshow("Pothole Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    picam2.stop()
    gps.close()
    cv2.destroyAllWindows()
    print("Cleanup complete")
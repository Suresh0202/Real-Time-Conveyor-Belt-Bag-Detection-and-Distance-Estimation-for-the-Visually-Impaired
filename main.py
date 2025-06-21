import cv2
import qrcode
import numpy as np
from ultralytics import YOLO
import pyttsx3

# === Load video file ===
cap = cv2.VideoCapture("IMG_3252.MOV")  # Replace with your filename

# === Load YOLO model ===
model = YOLO('yolov8n.pt')

# === Initialize text-to-speech ===
tts = pyttsx3.init()
spoken_ids = set()

# === Create QR with your name ===
qr_text = "Bag ID: SURESH_001"
qr_img = qrcode.make(qr_text).resize((70, 70))
qr_np = np.array(qr_img.convert('RGB'))
qr_cv = cv2.cvtColor(qr_np, cv2.COLOR_RGB2BGR)

# === Cache for missed detections ===
detection_cache = []
cache_frames = 10  # How many frames to keep showing missed boxes

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    new_boxes = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()

            if label in ['backpack', 'handbag', 'suitcase']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1

                # === Distance calculation ===
                FOCAL_LENGTH = 520
                KNOWN_WIDTH = 38
                distance_cm = (KNOWN_WIDTH * FOCAL_LENGTH) / width + 5

                # === Add to box cache for drawing later ===
                new_boxes.append(((x1, y1, x2, y2), label, distance_cm))

                # === Always show QR (no distance check) ===
                if y1 + 70 < frame.shape[0] and x1 + 70 < frame.shape[1]:
                    frame[y1:y1+70, x1:x1+70] = qr_cv

                # === Speak only once per unique bag ===
                uid = f"{label}_{x1}_{y1}"
                if uid not in spoken_ids:
                    tts.say(f"{label} detected at {int(distance_cm)} centimeters. QR says {qr_text}")
                    tts.runAndWait()
                    spoken_ids.add(uid)

    # === Update detection cache ===
    detection_cache.append(new_boxes)
    if len(detection_cache) > cache_frames:
        detection_cache.pop(0)

    # === Draw all cached boxes ===
    for cached_boxes in detection_cache:
        for (x1, y1, x2, y2), label, distance_cm in cached_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 204, 0), 2)
            label_text = f"{label.capitalize()} - {int(distance_cm)} cm"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2)

    # === Show final frame ===
    cv2.imshow("SURESH - Bag Detection with QR", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

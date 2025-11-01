import cv2
import mediapipe as mp
import time
import winsound

# Initialize face detection
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)

# Open webcam
cap = cv2.VideoCapture(0)

last_seen_time = time.time()
alert_delay = 2  # seconds before beep if face not detected

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    # Face observer status
    observer_color = (0, 255, 0)  # green = focused
    status_text = "Focused"

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                    int(bboxC.width * iw), int(bboxC.height * ih))
            cv2.rectangle(img, bbox, (0, 255, 0), 2)
            mp_draw.draw_detection(img, detection)
        last_seen_time = time.time()
    else:
        # Face not found → check if alert needed
        if time.time() - last_seen_time > alert_delay:
            observer_color = (0, 0, 255)  # red = distracted
            status_text = "Distracted"
            winsound.Beep(1000, 500)  # frequency, duration (ms)

    # Draw observer circle (bottom corner)
    cv2.circle(img, (50, 50), 30, observer_color, -1)
    cv2.putText(img, status_text, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, observer_color, 2)

    cv2.imshow("FocusGuard - Study Concentration Detector", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
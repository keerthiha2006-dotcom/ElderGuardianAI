import cv2
from fall_detector import PoseDetector

detector = PoseDetector()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = detector.process(frame)

    cv2.imshow("ElderGuardian AI - Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
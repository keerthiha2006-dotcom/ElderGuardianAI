import cv2


class EmotionDetector:
    """Detect common facial expressions and annotate them on a BGR frame."""

    def __init__(self):
        self._deepface = None
        self.error = None
        self.frame_number = 0
        self.last_detected = []
        self.detect_every = 30

    def _load_model(self):
        if self._deepface is not None or self.error is not None:
            return
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
        except Exception as error:
            self.error = str(error)

    def annotate(self, frame):
        self.frame_number += 1
        if self.frame_number % self.detect_every != 0 and self._deepface is not None:
            return frame, self.last_detected, "Emotion detection active"
        self._load_model()
        if self.error:
            return frame, [], "Emotion model unavailable. Install deepface."

        try:
            scale = min(1.0, 480 / max(frame.shape[:2]))
            analysis_frame = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
            faces = self._deepface.analyze(
                img_path=analysis_frame,
                actions=("emotion",),
                detector_backend="opencv",
                enforce_detection=False,
                silent=True,
            )
        except Exception as error:
            return frame, [], f"Emotion detection error: {error}"

        if isinstance(faces, dict):
            faces = [faces]
        detected = []
        for face in faces:
            region = face.get("region", {})
            x = int(max(0, region.get("x", 0)) / scale)
            y = int(max(0, region.get("y", 0)) / scale)
            width = int(region.get("w", 0) / scale)
            height = int(region.get("h", 0) / scale)
            emotion = face.get("dominant_emotion", "unknown").capitalize()
            score = float(face.get("emotion", {}).get(emotion.lower(), 0.0))
            detected.append({"emotion": emotion, "confidence": score})

            if width > 0 and height > 0:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 200, 0), 2)
                label = f"{emotion} {score:.0f}%"
                cv2.rectangle(frame, (x, max(0, y - 32)), (x + 170, y), (0, 200, 0), -1)
                cv2.putText(
                    frame, label, (x + 6, max(20, y - 9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                )
            self.last_detected = detected
        return frame, detected, "Emotion detection active"

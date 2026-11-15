import cv2
import mediapipe as mp
import math
import time


class PoseDetector:

    def __init__(self):

        try:
            self.mp_pose = mp.solutions.pose
            self.mp_draw = mp.solutions.drawing_utils
        except AttributeError:
            from mediapipe.python.solutions import drawing_utils, pose
            self.mp_pose = pose
            self.mp_draw = drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb)

        if results.pose_landmarks:

            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        return results

    def analyze(self, results):

        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark

        nose = lm[self.mp_pose.PoseLandmark.NOSE]

        left_shoulder = lm[
            self.mp_pose.PoseLandmark.LEFT_SHOULDER
        ]

        right_shoulder = lm[
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]

        left_hip = lm[
            self.mp_pose.PoseLandmark.LEFT_HIP
        ]

        right_hip = lm[
            self.mp_pose.PoseLandmark.RIGHT_HIP
        ]

        # Average shoulder position
        shoulder_x = (
            left_shoulder.x +
            right_shoulder.x
        ) / 2

        shoulder_y = (
            left_shoulder.y +
            right_shoulder.y
        ) / 2

        # Average hip position
        hip_x = (
            left_hip.x +
            right_hip.x
        ) / 2

        hip_y = (
            left_hip.y +
            right_hip.y
        ) / 2

        dx = hip_x - shoulder_x
        dy = hip_y - shoulder_y

        angle = abs(math.degrees(math.atan2(dy, dx)))

        # Normalize angle
        if angle > 90:
            angle = 180 - angle

        body_height = abs(hip_y - nose.y)

        body_width = abs(
            left_shoulder.x -
            right_shoulder.x
        )

        return {
            "torso_angle": angle,
            "body_height": body_height,
            "body_width": body_width,
            "shoulder_y": shoulder_y,
            "hip_y": hip_y,
        }


class FallMonitor:
    """Confirm a sudden, horizontal posture before raising an incident."""

    def __init__(self, angle_threshold=35, confirmation_seconds=1.0,
                 unconscious_seconds=10, cooldown_seconds=30):
        self.angle_threshold = angle_threshold
        self.confirmation_seconds = confirmation_seconds
        self.unconscious_seconds = unconscious_seconds
        self.cooldown_seconds = cooldown_seconds
        self._horizontal_since = None
        self._sudden_since = None
        self._last_alert = 0.0
        self._fall_latched = False
        self.previous = None

    def update(self, metrics):
        now = time.monotonic()
        if metrics is None:
            self._horizontal_since = None
            self._sudden_since = None
            self._fall_latched = False
            self.previous = None
            return {"fall_detected": False, "reason": "No pose detected"}

        angle = metrics["torso_angle"]
        sudden_change = (
            self.previous is not None
            and self.previous["torso_angle"] >= 60
            and angle <= self.angle_threshold
            and (
                metrics["hip_y"] - self.previous["hip_y"] >= 0.02
                or metrics["body_height"] <= self.previous["body_height"] * 0.8
            )
        )
        self.previous = metrics
        if sudden_change:
            self._sudden_since = now

        if angle <= self.angle_threshold:
            if self._horizontal_since is None:
                self._horizontal_since = now
        else:
            self._horizontal_since = None
            self._sudden_since = None

        horizontal_long_enough = (
            self._horizontal_since is not None
            and now - self._horizontal_since >= self.confirmation_seconds
        )
        detected = horizontal_long_enough and self._sudden_since is not None
        possible_unconsciousness = (
            self._horizontal_since is not None
            and now - self._horizontal_since >= self.unconscious_seconds
        )
        should_alert = (
            detected
            and not self._fall_latched
            and now - self._last_alert >= self.cooldown_seconds
        )
        if should_alert:
            self._last_alert = now
            self._fall_latched = True
        if angle > 60:
            self._fall_latched = False

        return {
            "fall_detected": detected,
            "should_alert": should_alert,
            "sudden_change": sudden_change,
            "possible_unconsciousness": possible_unconsciousness,
            "angle": angle,
            "reason": (
                "Possible unconsciousness: horizontal posture persisted"
                if possible_unconsciousness
                else "Horizontal posture after sudden movement"
                if detected else "Monitoring"
            ),
        }
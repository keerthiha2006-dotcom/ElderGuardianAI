"""
ElderGuardian AI — Intelligent Offline Elder Care Assistant
=============================================================
PHASE 1: Streamlit UI shell + live webcam feed via OpenCV.

The live loop runs MediaPipe Pose, displays torso angle and body height,
confirms sudden horizontal movement, and triggers local/phone alerts.

Run with:  streamlit run app.py
"""

import time
from queue import Empty, Queue
import cv2
import numpy as np
import streamlit as st
from alerts import VoiceListener, trigger_emergency
from emotion_detector import EmotionDetector
from fall_detector import FallMonitor, PoseDetector

# -----------------------------------------------------------------
# 1. PAGE CONFIG — must be the first Streamlit call in the script
# -----------------------------------------------------------------
st.set_page_config(
    page_title="ElderGuardian AI",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #17211b;
        --muted: #69756d;
        --line: #dfe7df;
        --paper: #f5f8f3;
        --panel: #ffffff;
        --mint: #dcefe4;
        --green: #1f7654;
        --amber: #c88724;
        --danger: #bd4c3e;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #edf4ee; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding: 2rem 1.25rem; }
    .block-container { max-width: 1440px; padding: 2rem 3rem 3rem; }
    h1, h2, h3, [data-testid="stMetricValue"] { font-family: 'Space Grotesk', 'Trebuchet MS', sans-serif; }
    h1 { font-size: clamp(2rem, 4vw, 3.55rem); letter-spacing: -1px; margin: 0; }
    h2 { font-size: 1.25rem; margin: 0; }
    p, label, .stCaption, [data-testid="stMarkdownContainer"] { font-family: 'DM Sans', 'Trebuchet MS', sans-serif; }
    .eyebrow { color: var(--green); font: 700 0.72rem 'DM Sans', sans-serif; letter-spacing: 1.8px; text-transform: uppercase; margin-bottom: .55rem; }
    .hero-copy { color: var(--muted); font-size: 1rem; margin: .45rem 0 1.5rem; }
    .live-pill { display: inline-flex; align-items: center; gap: .45rem; background: var(--mint); border: 1px solid #b8ddc6; border-radius: 999px; color: var(--green); font: 700 .76rem 'DM Sans', sans-serif; padding: .5rem .75rem; }
    .live-dot { width: 8px; height: 8px; border-radius: 50%; background: #2d9b69; box-shadow: 0 0 0 4px #c7e8d3; }
    .metric-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 1rem 1.15rem; min-height: 106px; }
    .metric-label { color: var(--muted); font: 600 .72rem 'DM Sans', sans-serif; letter-spacing: .7px; text-transform: uppercase; }
    .metric-value { color: var(--ink); font: 700 1.75rem 'Space Grotesk', sans-serif; margin-top: .45rem; }
    .metric-note { color: var(--muted); font: 400 .77rem 'DM Sans', sans-serif; margin-top: .2rem; }
    .feed-label { color: var(--muted); font: 700 .72rem 'DM Sans', sans-serif; letter-spacing: 1.2px; text-transform: uppercase; margin: 1.6rem 0 .6rem; }
    .incident-panel { background: #fff8ed; border: 1px solid #f0d7a9; border-left: 4px solid var(--amber); border-radius: 8px; padding: 1rem 1.1rem; }
    .incident-panel.alert { background: #fff0ed; border-color: #efc3bc; border-left-color: var(--danger); }
    .incident-title { color: var(--ink); font: 700 1rem 'Space Grotesk', sans-serif; }
    .incident-copy { color: var(--muted); font: .82rem 'DM Sans', sans-serif; margin-top: .25rem; }
    .sidebar-brand { color: var(--ink); font: 700 1.35rem 'Space Grotesk', sans-serif; }
    .sidebar-subtitle { color: var(--muted); font: .82rem 'DM Sans', sans-serif; line-height: 1.45; margin: .35rem 0 1.6rem; }
    div.stButton > button { border-radius: 6px; font-family: 'DM Sans', sans-serif; font-weight: 700; min-height: 2.7rem; }
    div.stButton > button[kind="primary"] { background: var(--green); border-color: var(--green); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------
# 2. SESSION STATE — Streamlit reruns the whole script on every
#    interaction, so anything that must persist across reruns
#    (like "is monitoring currently on?") lives in st.session_state.
# -----------------------------------------------------------------
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "detector" not in st.session_state:
    st.session_state.detector = PoseDetector()
if "fall_monitor" not in st.session_state:
    st.session_state.fall_monitor = FallMonitor()
if "alert_message" not in st.session_state:
    st.session_state.alert_message = "No active alerts"
if "voice_listener" not in st.session_state:
    st.session_state.voice_listener = None
if "emotion_detector" not in st.session_state:
    st.session_state.emotion_detector = EmotionDetector()
if "voice_events" not in st.session_state:
    st.session_state.voice_events = Queue()


def process_frame(frame: np.ndarray, emotion_enabled: bool):
    """
    Process one BGR frame and return the annotated frame plus metrics.
    """
    results = st.session_state.detector.process(frame)
    metrics = st.session_state.detector.analyze(results)
    event = st.session_state.fall_monitor.update(metrics)
    if emotion_enabled:
        frame, emotions, emotion_status = st.session_state.emotion_detector.annotate(frame)
    else:
        emotions, emotion_status = [], "Emotion detection paused for performance"

    if metrics:
        cv2.putText(
            frame, f"Torso angle: {metrics['torso_angle']:.1f} deg",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )
        cv2.putText(
            frame, f"Body height: {metrics['body_height']:.3f}",
            (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
        )

    if event.get("should_alert"):
        alert_status = trigger_emergency()
        st.session_state.alert_message = f"FALL DETECTED - {alert_status}"
    return frame, event, metrics, emotions, emotion_status


def handle_voice_alert(words):
    st.session_state.voice_events.put(words)


# -----------------------------------------------------------------
# 3. SIDEBAR — controls
# -----------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">ElderGuardian AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">A quiet second set of eyes for the people who matter.</div>', unsafe_allow_html=True)

    st.markdown("**Monitoring controls**")

    camera_index = st.number_input(
        "Camera index", min_value=0, max_value=5, value=0, step=1,
        help="0 is usually the built-in laptop webcam. Try 1 if you have an external cam.",
    )
    if st.button("Test emergency alert", width="stretch"):
        st.session_state.alert_message = trigger_emergency()
    st.caption("The test sounds a local alarm and updates the incident panel.")
    voice_enabled = st.checkbox("Listen for help / emergency", value=True)

    start_col, stop_col = st.columns(2)
    if start_col.button("▶ Start Monitoring", width="stretch"):
        st.session_state.monitoring = True
    if stop_col.button("⏹ Stop", width="stretch"):
        st.session_state.monitoring = False

    st.divider()
    st.markdown("**Detection layers**")
    emotion_enabled = st.checkbox("Enable emotion detection (slower)", value=False)
    st.caption("Expressions: happy, sad, angry, fear, surprise, disgust, neutral.")

    st.divider()
    st.caption("All pose and fall analysis runs locally. Voice recognition needs internet access.")

# -----------------------------------------------------------------
# 4. MAIN AREA — dashboard
# -----------------------------------------------------------------
header_left, header_right = st.columns([4, 1])
with header_left:
    st.markdown('<div class="eyebrow">Guardian station / live room view</div>', unsafe_allow_html=True)
    st.title("Care, with attention.")
    st.markdown('<div class="hero-copy">Real-time posture, movement, and emergency voice monitoring in one place.</div>', unsafe_allow_html=True)
with header_right:
    if st.session_state.monitoring:
        st.markdown('<div class="live-pill"><span class="live-dot"></span>MONITORING LIVE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="live-pill" style="background:#eef1ed;border-color:#d7ded7;color:#69756d"><span class="live-dot" style="background:#87948b;box-shadow:none"></span>STANDBY</div>', unsafe_allow_html=True)

status_placeholder = st.empty()
fps_placeholder = st.empty()
metric_angle = st.empty()
metric_height = st.empty()
metric_emotion = st.empty()
metric_camera = st.empty()

metric_cols = st.columns(4)
with metric_cols[0]: metric_angle.markdown('<div class="metric-card"><div class="metric-label">Torso angle</div><div class="metric-value">--</div><div class="metric-note">Waiting for pose</div></div>', unsafe_allow_html=True)
with metric_cols[1]: metric_height.markdown('<div class="metric-card"><div class="metric-label">Body height</div><div class="metric-value">--</div><div class="metric-note">Normalized frame value</div></div>', unsafe_allow_html=True)
with metric_cols[2]: metric_emotion.markdown('<div class="metric-card"><div class="metric-label">Faces detected</div><div class="metric-value">--</div><div class="metric-note">Emotion layer paused</div></div>', unsafe_allow_html=True)
with metric_cols[3]: metric_camera.markdown(f'<div class="metric-card"><div class="metric-label">Camera</div><div class="metric-value">#{int(camera_index)}</div><div class="metric-note">Local webcam input</div></div>', unsafe_allow_html=True)

st.markdown('<div class="feed-label">Live camera feed</div>', unsafe_allow_html=True)
video_placeholder = st.empty()

alert_placeholder = st.empty()
alert_placeholder.markdown(
    '<div class="incident-panel"><div class="incident-title">No active incident</div><div class="incident-copy">The guardian station is watching for sudden movement, horizontal posture, or an emergency voice cue.</div></div>',
    unsafe_allow_html=True,
)

if not st.session_state.monitoring:
    status_placeholder.info("Standby · Start monitoring from the control panel when the room is ready.")
    video_placeholder.info("Your live camera feed will appear here once monitoring starts.")

# -----------------------------------------------------------------
# 5. CAPTURE LOOP
#    Streamlit has no native "infinite live loop" widget, so the
#    standard pattern is: while a session-state flag is True, grab
#    a frame, show it, and re-check the flag every iteration. The
#    Start/Stop buttons above flip that flag on the next rerun.
# -----------------------------------------------------------------
if st.session_state.monitoring:
    cap = cv2.VideoCapture(int(camera_index))

    if not cap.isOpened():
        st.session_state.monitoring = False
        status_placeholder.error(
            f"❌ Could not open camera index {camera_index}. "
            "Check that no other app is using it, or try a different index."
        )
    else:
        status_placeholder.success("Monitoring active · camera connected")
        prev_time = time.time()

        while st.session_state.monitoring:
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("❌ Lost connection to camera.")
                break

            # Mirror the feed so it feels natural (like a mirror, not a recording)
            frame = cv2.flip(frame, 1)

            if voice_enabled and st.session_state.voice_listener is None:
                st.session_state.voice_listener = VoiceListener(handle_voice_alert)
                st.session_state.voice_listener.start()

            # Run pose estimation, angle tracking, and fall confirmation.
            try:
                voice_words = st.session_state.voice_events.get_nowait()
                alert_status = trigger_emergency()
                st.session_state.alert_message = f"VOICE ALERT: {voice_words} - {alert_status}"
                alert_placeholder.markdown(f'<div class="incident-panel alert"><div class="incident-title">Voice emergency detected</div><div class="incident-copy">{st.session_state.alert_message}</div></div>', unsafe_allow_html=True)
            except Empty:
                pass
            frame, event, metrics, emotions, emotion_status = process_frame(
                frame, emotion_enabled
            )
            if emotions and st.session_state.frame_count % 10 == 0:
                status_placeholder.info(
                    " | ".join(
                        f"Face {index + 1}: {item['emotion']} ({item['confidence']:.0f}%)"
                        for index, item in enumerate(emotions)
                    )
                )
            if event.get("fall_detected"):
                if not event.get("should_alert"):
                    st.session_state.alert_message = f"FALL DETECTED - {event.get('reason', 'possible fall')}"
                alert_placeholder.markdown(f'<div class="incident-panel alert"><div class="incident-title">Possible fall detected</div><div class="incident-copy">{st.session_state.alert_message}</div></div>', unsafe_allow_html=True)
                status_placeholder.error("Possible fall: " + event["reason"])
            elif metrics and st.session_state.frame_count % 10 == 0:
                status_placeholder.success(
                    f"Monitoring active · torso angle {metrics['torso_angle']:.1f} deg"
                )

            if metrics and st.session_state.frame_count % 5 == 0:
                metric_angle.markdown(f'<div class="metric-card"><div class="metric-label">Torso angle</div><div class="metric-value">{metrics["torso_angle"]:.1f}°</div><div class="metric-note">Fall threshold: ≤ {st.session_state.fall_monitor.angle_threshold}°</div></div>', unsafe_allow_html=True)
                metric_height.markdown(f'<div class="metric-card"><div class="metric-label">Body height</div><div class="metric-value">{metrics["body_height"]:.3f}</div><div class="metric-note">Normalized frame value</div></div>', unsafe_allow_html=True)
            if st.session_state.frame_count % 5 == 0:
                metric_emotion.markdown(f'<div class="metric-card"><div class="metric-label">Faces detected</div><div class="metric-value">{len(emotions)}</div><div class="metric-note">{"Emotion layer active" if emotion_enabled else "Emotion layer paused"}</div></div>', unsafe_allow_html=True)

            if emotion_enabled and not emotions and emotion_status != "Emotion detection active":
                alert_placeholder.markdown(f'<div class="incident-panel"><div class="incident-title">Emotion layer unavailable</div><div class="incident-copy">{emotion_status}</div></div>', unsafe_allow_html=True)

            if voice_enabled and st.session_state.voice_listener and not st.session_state.voice_listener.available:
                st.session_state.alert_message = (
                    "Voice listener unavailable: install SpeechRecognition and PyAudio"
                )

            # OpenCV uses BGR; Streamlit's st.image expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", width="stretch")

            # Lightweight FPS readout so you can see it's really live
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            st.session_state.frame_count += 1
            fps_placeholder.caption(f"~{fps:.1f} FPS · frame #{st.session_state.frame_count}")

            # Streamlit reruns the script when session_state changes
            # elsewhere (e.g. Stop button), so this loop re-checks
            # st.session_state.monitoring on every iteration and will
            # exit cleanly on the next rerun triggered by that click.

        cap.release()
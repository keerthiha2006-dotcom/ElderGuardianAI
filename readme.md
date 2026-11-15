# ElderGuardian AI

## Run

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Enter a guardian or helpline number in the sidebar. The app confirms a fall
when the torso angle changes suddenly and remains at or below 45 degrees for
0.8 seconds. Ten seconds of continued horizontal posture is labeled possible
unconsciousness. A local Windows beep and on-screen alert are immediate.

## Local alerts

Press **Test emergency alert** to test the on-screen message and computer alarm.
Fall detection and spoken emergency words show the same alert in the dashboard.

## Voice emergency detection

Enable **Listen for help / emergency** and allow microphone access. The
listener reacts to speech containing `help`, `emergency`, or `fall`. Speech
recognition uses Google's recognition endpoint, so it needs internet access;
the pose and fall detection remain local.

This is an assistive prototype, not a medical device. Test camera position,
lighting, microphone access, and the alert number before relying on it.

## Train the supplied dataset

From this project folder, run:

```powershell
python -m pip install -r requirements.txt
python train_fall_model.py C:\Users\owner\OneDrive\Desktop\urfall-cam0-falls.csv
```

This creates `models/fall_model.joblib` and `models/fall_model_report.json`.
The training script uses video-level splitting, so frames from the same video
cannot appear in both training and testing. The CSV model expects its original
eight numeric features; the live webcam uses MediaPipe landmarks and its
temporal detector, because those are different feature spaces.

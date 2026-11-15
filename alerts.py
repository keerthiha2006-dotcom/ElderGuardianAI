import platform
def trigger_emergency():
    """Sound the local alarm and return a visible alert message."""
    emergency_alert()
    return "Emergency alarm sounded"


class VoiceListener:
    """Listen for emergency words when SpeechRecognition is installed."""

    def __init__(self, callback):
        self.callback = callback
        self.recognizer = None
        self.microphone = None
        self.error = None
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
        except Exception as error:
            self.error = str(error)

    @property
    def available(self):
        return self.recognizer is not None and self.microphone is not None

    def start(self):
        if not self.available:
            return

        def on_audio(recognizer, audio):
            try:
                text = recognizer.recognize_google(audio).lower()
                if any(word in text for word in ("help", "emergency", "fall")):
                    self.callback(text)
            except Exception:
                pass

        self.recognizer.listen_in_background(self.microphone, on_audio)


def emergency_alert():

    print("\n")
    print("=" * 50)
    print("🚨 EMERGENCY ALERT")
    print("Possible fall + prolonged inactivity detected")
    print("=" * 50)

    # Simple computer beep
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 800)
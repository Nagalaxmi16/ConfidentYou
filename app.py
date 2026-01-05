import streamlit as st
import cv2
import mediapipe as mp
import av
import speech_recognition as sr
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import time
import pandas as pd
import random

# --- PAGE CONFIG ---
st.set_page_config(page_title="ConfidentYou", layout="wide")

# --- INITIALIZE SESSION STATE ---
if "filler_count" not in st.session_state:
    st.session_state.filler_count = 0
if "score_history" not in st.session_state:
    st.session_state.score_history = []
if "current_question" not in st.session_state:
    st.session_state.current_question = "Click 'Next Question' to start!"

# --- QUESTION BANK ---
QUESTIONS = [
    "Tell me about yourself.",
    "What are your greatest strengths and weaknesses?",
    "Why do you want to work for this company?",
    "Describe a challenge you faced and how you overcame it.",
    "Where do you see yourself in five years?",
]

# --- MEDIAPIPE SETUP ---
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection


class VideoProcessor(VideoProcessorBase):
    eye_contact = False
    posture_good = False

    def __init__(self):
        self.face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)
        self.pose_detector = mp_pose.Pose(min_detection_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_results = self.face_detector.process(rgb)
        VideoProcessor.eye_contact = True if face_results.detections else False

        pose_results = self.pose_detector.process(rgb)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            VideoProcessor.posture_good = abs(left_shoulder - right_shoulder) < 0.05

        color_eye = (0, 255, 0) if VideoProcessor.eye_contact else (0, 0, 255)
        cv2.putText(
            img,
            f"Eye Contact: {'Good' if VideoProcessor.eye_contact else 'Poor'}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_eye,
            2,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def draw_3d_coach():
    anim = "Idle"
    if not VideoProcessor.eye_contact:
        anim = "Punched"
    elif not VideoProcessor.posture_good:
        anim = "Wave"

    model_html = f"""
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    <div style="position: relative;">
        <model-viewer src="https://modelviewer.dev/shared-assets/models/RobotExpressive.glb" 
                      autoplay animation-name="{anim}"
                      style="width: 100%; height: 300px; background-color: #0e1117;"></model-viewer>
        <div style="position: absolute; top: 10px; left: 10px; right: 10px; background: rgba(255,255,255,0.9); 
                    padding: 10px; border-radius: 10px; border: 2px solid #0078ff; color: black; font-weight: bold; font-family: sans-serif;">
            Coach: "{st.session_state.current_question}"
        </div>
    </div>
    """
    components.html(model_html, height=350)


st.title("ConfidentYou – AI Mock Interview")

with st.sidebar:
    st.header("🤖 Interviewer")
    draw_3d_coach()

    if st.button("Next Question ⏭️"):
        st.session_state.current_question = random.choice(QUESTIONS)
        st.rerun()

    st.markdown("---")
    if st.button("Record Answer (5s)"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.toast("Recording...")
            try:
                audio = r.listen(source, phrase_time_limit=5)
                text = r.recognize_google(audio).lower()
                fillers = ["um", "uh", "like", "actually"]
                st.session_state.filler_count += sum(
                    1 for w in text.split() if w in fillers
                )
                st.info(f"You said: {text}")
            except:
                st.error("Could not process audio.")

col1, col2 = st.columns([2, 1])

with col1:
    webrtc_streamer(
        key="coach",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

with col2:
    st.subheader("Live Assessment")
    st.metric("Eye Contact", "Good" if VideoProcessor.eye_contact else "Poor")
    st.metric("Posture", "Proper" if VideoProcessor.posture_good else "Fix Shoulders")
    st.metric("Filler Words", st.session_state.filler_count)

    current_score = (
        100 if VideoProcessor.eye_contact and VideoProcessor.posture_good else 50
    )
    st.session_state.score_history.append(current_score)

    if st.button("Finish & Evaluate"):
        st.header("📊 Performance Review")
        if st.session_state.score_history:
            chart_data = pd.DataFrame(
                st.session_state.score_history, columns=["Confidence"]
            )
            st.line_chart(chart_data)

        final_score = max(
            0,
            (sum(st.session_state.score_history) / len(st.session_state.score_history))
            - (st.session_state.filler_count * 5),
        )
        st.success(f"Final Interview Score: {int(final_score)}%")
        st.stop()

time.sleep(0.5)
st.rerun()

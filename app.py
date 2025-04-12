import streamlit as st
import google.generativeai as genai
import os
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import tempfile
from io import BytesIO
import base64
from dotenv import load_dotenv
from pathlib import Path
import time
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import queue

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Voice Assistant", page_icon="🤖", layout="wide")

# Initialize session state variables
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'gemini_chat' not in st.session_state:
    st.session_state.gemini_chat = None
if 'api_key_status' not in st.session_state:
    st.session_state.api_key_status = False
if 'audio_counter' not in st.session_state:
    st.session_state.audio_counter = 0
if 'recording_triggered' not in st.session_state:
    st.session_state.recording_triggered = False

# Backend API key handling
def get_api_key():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets["SECRET"]
        except:
            pass
    if not api_key:
        cred_paths = [
            Path('credentials.json'),
            Path('config/credentials.json'),
            Path('config/gemini_credentials.json')
        ]
        for path in cred_paths:
            if path.exists():
                try:
                    import json
                    with open(path, 'r') as f:
                        creds = json.load(f)
                        if 'api_key' in creds:
                            api_key = creds['api_key']
                            break
                except:
                    pass
    return api_key

# Setup Gemini
def setup_gemini():
    api_key = get_api_key()
    if not api_key:
        st.error("API key not found. Please add your Gemini API key.")
        st.session_state.api_key_status = False
        return None
    try:
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 1024,
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        chat = model.start_chat(history=[])
        st.session_state.api_key_status = True
        return chat
    except Exception as e:
        st.error(f"Gemini API error: {str(e)}")
        st.session_state.api_key_status = False
        return None

# Convert AI response to speech
def create_autoplay_audio(ai_response):
    tts = gTTS(text=ai_response, lang='en')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    audio_bytes = mp3_fp.read()
    b64 = base64.b64encode(audio_bytes).decode()
    st.session_state.audio_counter += 1
    audio_html = f"""
    <audio id="audio-{st.session_state.audio_counter}" autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    <script>
        document.getElementById("audio-{st.session_state.audio_counter}").play();
    </script>
    """
    return audio_html

# Process audio with Gemini
def process_with_gemini(audio_data, chat):
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_data) as source:
            audio = recognizer.record(source)
        user_input = recognizer.recognize_google(audio)
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        response = chat.send_message(user_input)
        ai_response = response.text
        st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})
        return user_input, ai_response
    except Exception as e:
        st.error(f"Speech recognition error: {str(e)}")
        return None, f"Error: {str(e)}"

# WebRTC audio recording processor
audio_queue = queue.Queue()

class AudioProcessor:
    def __init__(self):
        self.recorded_frames = []

    def recv(self, frame: av.AudioFrame):
        audio_queue.put(frame.to_ndarray())
        return frame

# WebRTC audio recording logic
def record_audio_streamlit(duration_sec=5):
    webrtc_ctx = webrtc_streamer(
        key="speech",
        mode=WebRtcMode.SENDONLY,
        in_audio=True,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    if webrtc_ctx.state.playing and not st.session_state.recording_triggered:
        st.session_state.recording_triggered = True
        with st.spinner("Recording... Please speak now."):
            time.sleep(duration_sec)
            webrtc_ctx.stop()
            audio_frames = []
            while not audio_queue.empty():
                frame = audio_queue.get()
                audio_frames.append(frame)

            if audio_frames:
                audio = np.concatenate(audio_frames, axis=1).T
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                sf.write(temp_file.name, audio, 48000)
                st.session_state.audio_data = temp_file.name
                return temp_file.name
    return None

# UI Layout
st.title("🤖 Voice Assistant")
st.markdown("Talk to an AI assistant using your voice")

if st.session_state.gemini_chat is None:
    st.session_state.gemini_chat = setup_gemini()

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🎙️ Voice Input")

    if st.session_state.api_key_status:
        st.success("API Key verified ✓")
    else:
        st.error("API Key missing or invalid ✗")

    audio_output = st.empty()

    if st.button("🎤 Press to Record", use_container_width=True, disabled=not st.session_state.api_key_status):
        st.session_state.recording_triggered = False
        audio_file = record_audio_streamlit(duration_sec=5)
        if audio_file and st.session_state.gemini_chat:
            st.info("Processing your voice...")
            user_text, ai_text = process_with_gemini(audio_file, st.session_state.gemini_chat)
            if ai_text:
                st.success("Assistant is responding...")
                audio_html = create_autoplay_audio(ai_text)
                audio_output.markdown(audio_html, unsafe_allow_html=True)

with col2:
    st.subheader("🗨️ Conversation History")
    if not st.session_state.conversation_history:
        st.info("Your conversation will appear here")
    else:
        for message in st.session_state.conversation_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI:** {message['content']}")
            st.divider()

with st.expander("Settings & Information"):
    st.markdown("""
    ### How to use this Voice Assistant
    1. Click the "Press to Record" button
    2. Speak clearly into your browser microphone
    3. Wait for the AI to respond with voice

    ### Requirements
    - Microphone access allowed in browser
    - Google Gemini API key (set via environment or secrets)

    ### Tech Stack
    - Google Gemini for NLP
    - SpeechRecognition for STT
    - gTTS for TTS
    - Streamlit + streamlit-webrtc for UI and audio recording
    """)
    if st.button("Reset Conversation"):
        st.session_state.conversation_history = []
        st.session_state.gemini_chat = None
        st.session_state.audio_counter = 0
        st.experimental_rerun()

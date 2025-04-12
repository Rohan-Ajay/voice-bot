import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import google.generativeai as genai
import os
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import base64
import tempfile
import soundfile as sf
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Voice Assistant", page_icon="🎙️", layout="wide")

# Session state initialization
for var in ["conversation_history", "recording", "audio_data", "gemini_chat", "api_key_status", "audio_counter"]:
    if var not in st.session_state:
        st.session_state[var] = [] if var == "conversation_history" else False if var == "recording" else None if var in ["audio_data", "gemini_chat"] else 0

# --- API Key Retrieval ---
def get_api_key():
    # Check Streamlit secrets
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if api_key:
        return api_key

    # Check credentials files
    cred_paths = ["credentials.json", "config/credentials.json", "config/gemini_credentials.json"]
    for path in cred_paths:
        if Path(path).exists():
            try:
                import json
                with open(path, 'r') as f:
                    creds = json.load(f)
                    return creds.get("api_key", None)
            except:
                pass
    return None

# --- Gemini Setup ---
def setup_gemini():
    api_key = get_api_key()
    if not api_key:
        st.error("Gemini API key not found. Add it to secrets or credentials.json.")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config={"temperature": 0.7, "top_p": 0.95, "top_k": 64, "max_output_tokens": 1024},
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
        )
        chat = model.start_chat(history=[])
        st.session_state.api_key_status = True
        return chat
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {e}")
        st.session_state.api_key_status = False
        return None

# --- Convert AI text to auto-playing audio ---
def create_autoplay_audio(text):
    tts = gTTS(text=text, lang='en')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    b64 = base64.b64encode(mp3_fp.read()).decode()
    st.session_state.audio_counter += 1
    return f"""
    <audio id="audio-{st.session_state.audio_counter}" autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    <script>document.getElementById("audio-{st.session_state.audio_counter}").play();</script>
    """

# --- Process voice with Gemini ---
def process_with_gemini(audio_path, chat):
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        user_input = recognizer.recognize_google(audio)
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        response = chat.send_message(user_input)
        ai_text = response.text
        st.session_state.conversation_history.append({"role": "assistant", "content": ai_text})
        return user_input, ai_text
    except Exception as e:
        st.error(f"Error processing speech: {e}")
        return None, None

# --- Save audio buffer to file ---
def save_audio_from_buffer(audio_buffer):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        sf.write(tmpfile.name, audio_buffer.to_ndarray(), audio_buffer.samplerate)
        return tmpfile.name

# --- Title & API Check ---
st.title("🎙️ Voice Assistant")
st.markdown("Speak to an AI assistant powered by Google Gemini.")

if st.session_state.gemini_chat is None:
    st.session_state.gemini_chat = setup_gemini()

if st.session_state.api_key_status:
    st.success("✅ API key valid")
else:
    st.warning("⚠️ Missing or invalid Gemini API key")

# --- WebRTC Audio Streamer ---
st.subheader("🎧 Speak Now")
webrtc_ctx = webrtc_streamer(
    key="streamer",
    mode=WebRtcMode.SENDONLY,
    client_settings=ClientSettings(media_stream_constraints={"audio": True, "video": False}),
)

# --- Process Button ---
if webrtc_ctx.audio_receiver:
    audio_buffer = webrtc_ctx.audio_receiver.get_new_audio_frames()
    if len(audio_buffer) > 0:
        st.session_state.recording = True
        audio_data = audio_buffer[0]
        audio_path = save_audio_from_buffer(audio_data)
        st.session_state.audio_data = audio_path

        st.info("Processing your voice...")
        user_text, ai_text = process_with_gemini(audio_path, st.session_state.gemini_chat)
        if ai_text:
            st.markdown(create_autoplay_audio(ai_text), unsafe_allow_html=True)

# --- Chat History ---
st.subheader("💬 Conversation History")
for msg in st.session_state.conversation_history:
    st.markdown(f"**{'You' if msg['role']=='user' else 'AI'}:** {msg['content']}")
    st.divider()

# --- Reset Button ---
with st.expander("⚙️ Settings"):
    if st.button("🔄 Reset Conversation"):
        st.session_state.conversation_history = []
        st.session_state.gemini_chat = setup_gemini()
        st.session_state.audio_counter = 0
        st.rerun()

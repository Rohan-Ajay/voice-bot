import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import google.generativeai as genai
import os
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import base64
import tempfile
import wave

# Set Streamlit page config
st.set_page_config(page_title="Voice Assistant", page_icon="🎤", layout="wide")

# Session state initialization
for key in ['conversation_history', 'gemini_chat', 'audio_counter']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'conversation_history' else 0 if key == 'audio_counter' else None

# Get API key from environment or secrets
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except:
        return os.getenv("GEMINI_API_KEY")

# Set up Gemini model
def setup_gemini():
    api_key = get_api_key()
    if not api_key:
        st.error("API Key not found.")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config={"temperature": 0.7, "top_p": 0.95, "top_k": 40, "max_output_tokens": 1024},
            safety_settings=[{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_LOW_AND_ABOVE"}],
        )
        return model.start_chat(history=[])
    except Exception as e:
        st.error(f"Gemini setup failed: {e}")
        return None

# Text-to-Speech playback in browser
def create_autoplay_audio(response_text):
    tts = gTTS(text=response_text, lang='en')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    b64 = base64.b64encode(mp3_fp.read()).decode()
    st.session_state.audio_counter += 1
    return f"""
    <audio id="audio-{st.session_state.audio_counter}" autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    <script>
        document.getElementById("audio-{st.session_state.audio_counter}").play();
    </script>
    """

# Audio Processor Class
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.frames = []

    def recv(self, frame):
        self.frames.append(frame.to_ndarray().flatten().tobytes())
        return frame

    def get_audio_bytes(self):
        return b''.join(self.frames)

# --- MAIN UI ---
st.title("🎤 Voice Assistant")
st.markdown("Talk to an AI using your mic")

# Initialize Gemini
if st.session_state.gemini_chat is None:
    st.session_state.gemini_chat = setup_gemini()

# Set up columns
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Speak to the Assistant")

    # Start WebRTC streamer
    webrtc_streamer(
        key="audio",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {
                    "urls": ["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443"],
                    "username": "openrelayproject",
                    "credential": "openrelayproject"
                },
            ]
        }
    )


    audio_output = st.empty()

    # Button to stop and process audio
    if st.button("🛑 Process My Voice", disabled=not webrtc_ctx.state.playing):
        if webrtc_ctx.audio_processor:
            st.info("Converting speech...")
            audio_bytes = webrtc_ctx.audio_processor.get_audio_bytes()
            
            # Save to .wav for speech recognition
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(audio_bytes)

            try:
                # Transcribe with SpeechRecognition
                recognizer = sr.Recognizer()
                with sr.AudioFile(temp_path) as source:
                    audio = recognizer.record(source)
                user_input = recognizer.recognize_google(audio)

                # Send to Gemini
                response = st.session_state.gemini_chat.send_message(user_input)
                ai_response = response.text

                # Store history
                st.session_state.conversation_history.append({"role": "user", "content": user_input})
                st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})

                # Speak back
                audio_html = create_autoplay_audio(ai_response)
                audio_output.markdown(audio_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Speech error: {e}")

with col2:
    st.subheader("Conversation History")
    if not st.session_state.conversation_history:
        st.info("Start talking to see messages here.")
    else:
        for msg in st.session_state.conversation_history:
            role = "You" if msg["role"] == "user" else "AI"
            st.markdown(f"**{role}:** {msg['content']}")
            st.divider()

# Reset option
with st.expander("⚙️ Settings & Help"):
    if st.button("🔁 Reset Conversation"):
        st.session_state.conversation_history = []
        st.session_state.gemini_chat = None
        st.experimental_rerun()
    st.markdown("""
    - Make sure you allow mic permissions.
    - Click "Stop and Process" after speaking.
    - This app uses:
      - `streamlit-webrtc` for mic input
      - `SpeechRecognition` to transcribe
      - Google Gemini API for responses
      - `gTTS` to speak back
    """)


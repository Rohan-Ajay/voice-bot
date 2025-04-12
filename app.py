import streamlit as st
from streamlit_mic_recorder import mic_recorder
import google.generativeai as genai
import os
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import base64
import tempfile
from pydub import AudioSegment

# Page setup
st.set_page_config(page_title="Voice Assistant", page_icon="🎤", layout="wide")

# Init session state
for key in ['conversation_history', 'gemini_chat', 'audio_counter']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'conversation_history' else 0 if key == 'audio_counter' else None

# Get Gemini API key
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except:
        return os.getenv("GEMINI_API_KEY")

# Set up Gemini
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

# TTS to HTML player
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
    <script>
        document.getElementById("audio-{st.session_state.audio_counter}").play();
    </script>
    """

# Initialize Gemini chat
if st.session_state.gemini_chat is None:
    st.session_state.gemini_chat = setup_gemini()

# Layout
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🎙️ Record your voice")

    audio_data = mic_recorder(start_prompt="Start talking", stop_prompt="Click to stop", key="recorder")

    if audio_data and audio_data["bytes"]:
        st.audio(audio_data["bytes"], format="audio/webm")

        try:
            # Save and convert audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
                f.write(audio_data["bytes"])
                webm_path = f.name

            audio = AudioSegment.from_file(webm_path, format="webm")
            audio = audio.set_channels(1).set_frame_rate(16000)

            wav_path = webm_path.replace(".webm", ".wav")
            audio.export(wav_path, format="wav")

            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)

            user_input = recognizer.recognize_google(audio_data)

            # Ask Gemini
            response = st.session_state.gemini_chat.send_message(user_input)
            ai_response = response.text

            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})

            # Play response
            st.markdown(create_autoplay_audio(ai_response), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

with col2:
    st.subheader("🧠 Conversation History")
    if not st.session_state.conversation_history:
        st.info("Start recording to see messages here.")
    else:
        for msg in st.session_state.conversation_history:
            role = "You" if msg["role"] == "user" else "AI"
            st.markdown(f"**{role}:** {msg['content']}")
            st.divider()

with st.expander("⚙️ Settings & Help"):
    if st.button("🔁 Reset Conversation"):
        st.session_state.conversation_history = []
        st.session_state.gemini_chat = None
        st.experimental_rerun()
    st.markdown("""
    - Record your voice using the built-in browser mic.
    - AI will transcribe, respond, and speak back.
    - Powered by:
        - `mic-recorder-streamlit`
        - Google Gemini API
        - `SpeechRecognition` + `gTTS`
    """)


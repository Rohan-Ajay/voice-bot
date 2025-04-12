import streamlit as st
from st_audiorec import st_audiorec
import google.generativeai as genai
import os
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import base64
import tempfile
import wave
from pydub import AudioSegment

# Set Streamlit page config
st.set_page_config(page_title="Voice Assistant", page_icon="🎤", layout="wide")

# Initialize session state
for key in ['conversation_history', 'gemini_chat', 'audio_counter']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'conversation_history' else 0 if key == 'audio_counter' else None

# Get API key from environment or secrets
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

# TTS playback
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

# Main UI
st.title("🎤 Voice Assistant")
st.markdown("Record your voice, let the AI respond with speech!")

# Init Gemini
if st.session_state.gemini_chat is None:
    st.session_state.gemini_chat = setup_gemini()

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🎙️ Record Your Voice")
    wav_audio_data = st_audiorec()

    if wav_audio_data:
        st.audio(wav_audio_data, format='audio/wav')
        st.info("Transcribing and sending to Gemini...")

        try:
            # Save uploaded data to WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                tmpfile.write(wav_audio_data)
                tmp_path = tmpfile.name

            # Convert to 16-bit PCM WAV using pydub
            audio = AudioSegment.from_file(tmp_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            processed_path = tmp_path.replace(".wav", "_processed.wav")
            audio.export(processed_path, format="wav")

            # Transcribe with SpeechRecognition
            recognizer = sr.Recognizer()
            with sr.AudioFile(processed_path) as source:
                audio_data = recognizer.record(source)
            user_input = recognizer.recognize_google(audio_data)

            # Gemini response
            response = st.session_state.gemini_chat.send_message(user_input)
            ai_response = response.text

            # Store history
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})

            # Speak it back
            audio_html = create_autoplay_audio(ai_response)
            st.markdown(audio_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during speech processing: {e}")

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
    - Press the mic button to record your voice.
    - After recording, the assistant will respond.
    - Powered by:
      - `streamlit-audio-recorder` for mic input
      - `SpeechRecognition` for transcription
      - Gemini API for replies
      - `gTTS` for speech back
    """)


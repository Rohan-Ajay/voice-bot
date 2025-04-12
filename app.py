import streamlit as st
import google.generativeai as genai
import os
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import tempfile
from io import BytesIO
import base64
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv 
from pathlib import Path
import time

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Voice Assistant", page_icon="🤖", layout="wide")

# Initialize session state variables
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'gemini_chat' not in st.session_state:
    st.session_state.gemini_chat = None
if 'api_key_status' not in st.session_state:
    st.session_state.api_key_status = False
if 'audio_counter' not in st.session_state:
    st.session_state.audio_counter = 0

# Backend API key handling
def get_api_key():
    # First check environment variables
    api_key = st.secrets["GEMINI_API_KEY"]
    
    print(api_key)

    # Then check secrets file
    if not api_key:
        try:
            api_key = st.secrets["SECRET"]
        except:
            pass
    
    
    # Check for credentials file
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

# Function to set up Gemini model
def setup_gemini():
    api_key = get_api_key()
    st.write(api_key)
    
    if not api_key:
        st.error("API key not found. Please add your Gemini API key to environment variables, .env file, or credentials.json")
        st.session_state.api_key_status = False
        return None
    
    try:
        genai.configure(api_key=api_key)
        
        # Set up the model
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
        st.error(f"Error initializing Gemini API: {str(e)}")
        st.session_state.api_key_status = False
        return None

# Function to create auto-playing audio
def create_autoplay_audio(ai_response):
    # Convert response to speech
    tts = gTTS(text=ai_response, lang='en')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    
    # Get audio bytes and encode to base64
    audio_bytes = mp3_fp.read()
    b64 = base64.b64encode(audio_bytes).decode()
    
    # Increment the audio counter to force re-rendering
    st.session_state.audio_counter += 1
    
    # Create HTML with autoplay
    # Use a unique key in the element ID to force re-rendering
    audio_html = f"""
    <audio id="audio-{st.session_state.audio_counter}" autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    <script>
        // Force play the audio
        document.getElementById("audio-{st.session_state.audio_counter}").play();
    </script>
    """
    return audio_html

# Function to process voice input with Gemini and generate speech response
def process_with_gemini(audio_data, chat):
    try:
        # Convert audio to text using speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_data) as source:
            audio = recognizer.record(source)
        
        user_input = recognizer.recognize_google(audio)
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        
        # Process with Gemini
        response = chat.send_message(user_input)
        ai_response = response.text
        st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})
        
        return user_input, ai_response
        
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None, f"Error: {str(e)}"

# Function to record audio
def record_audio(duration=5, fs=16000):
    st.session_state.recording = True
    st.session_state.audio_data = None
    
    # Record audio
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    
    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_file.name, recording, fs)
    
    st.session_state.audio_data = temp_file.name
    st.session_state.recording = False
    
    return temp_file.name

# Main app UI
st.title("🤖 Voice Assistant")
st.markdown("Talk to an AI assistant")

# Initialize Gemini chat if not already initialized
if st.session_state.gemini_chat is None:
    st.session_state.gemini_chat = setup_gemini()

# Create columns for better layout
col1, col2 = st.columns([3, 2])

with col1:
    # Voice input section
    st.subheader("Speak to the Assistant")
    
    # API key status indicator
    if st.session_state.api_key_status:
        st.success("API Key verified ✓")
    else:
        st.error("API Key missing or invalid ✗")
        st.info("Please add your Gemini API key to environment variables, .env file, or credentials.json")
    
    # Audio output container (will be updated with auto-playing audio)
    audio_output = st.empty()
    
    # Record button - only enabled if API key is valid
    record_button = st.button("🎤 Press to Talk", key="record_button", use_container_width=True, disabled=not st.session_state.api_key_status)
    
    # Display recording status
    recording_status = st.empty()
    
    if record_button:
        recording_status.info("Listening... Please speak now")
        audio_file = record_audio(duration=5)
        
        if audio_file and st.session_state.gemini_chat:
            recording_status.info("Processing your request...")
            user_text, ai_text = process_with_gemini(audio_file, st.session_state.gemini_chat)
            
            if ai_text:
                recording_status.success("Assistant is responding")
                # Generate HTML for auto-playing audio
                audio_html = create_autoplay_audio(ai_text)
                # Update the audio container with auto-playing audio
                audio_output.markdown(audio_html, unsafe_allow_html=True)
        else:
            recording_status.error("Unable to process audio. Please try again.")

with col2:
    # Display conversation history
    st.subheader("Conversation History")
    
    conversation_container = st.container()
    with conversation_container:
        if not st.session_state.conversation_history:
            st.info("Your conversation will appear here")
        else:
            for message in st.session_state.conversation_history:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**AI:** {message['content']}")
                st.divider()

# Settings and info in expander
with st.expander("Settings & Information"):
    st.markdown("""
    ### How to use this Voice Assistant
    1. Click the "Press to Talk" button
    2. Speak clearly into your microphone
    3. Wait for the AI to process and respond with voice (automatic playback)
    
    ### Requirements
    - Microphone access must be allowed in your browser
    - A valid Google Gemini API key is required (configured in backend)
    
    ### Technical Information
    This application uses:
    - Google's Gemini AI model for natural language processing
    - SpeechRecognition for converting speech to text
    - gTTS (Google Text-to-Speech) for converting text back to speech
    - Streamlit for the web interface
    """)
    
    # Reset conversation button
    if st.button("Reset Conversation"):
        st.session_state.conversation_history = []
        st.session_state.gemini_chat = None
        st.session_state.audio_counter = 0
        st.experimental_rerun()


import streamlit as st
import os
from PIL import Image
import sounddevice as sd
import numpy as np
import queue
import threading
import json
from faster_whisper import WhisperModel
import code1.testspacy as testspacy
import time

# Set page config
st.set_page_config(page_title="Interactive Learning Assistant", page_icon="🎓", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .big-font {
        font-size: 24px !important;
    }
    .medium-font {
        font-size: 18px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the Whisper model
@st.cache_resource
def load_whisper_model():
    return WhisperModel("small", device="cpu", compute_type="int8")

model = load_whisper_model()

# Initialize spaCy for keyword extraction
@st.cache_resource
def load_spacy_model():
    return testspacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Create queues
audio_queue = queue.Queue()
gui_queue = queue.Queue()

# Define audio stream parameters
samplerate = 16000
duration = 2

# Function to load keyword-image mapping
@st.cache_data
def load_keyword_image_map(config_file):
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"The file {config_file} was not found.")
    except json.JSONDecodeError:
        st.error(f"The file {config_file} is not a valid JSON.")
    return {}

keyword_image_map = load_keyword_image_map('keywords.json')

# Function to extract keywords from text using spaCy
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
    return keywords

# Callback function to capture audio input
def callback(indata, frames, time, status):
    if status:
        st.error(status)
    audio_queue.put(indata)

# Function to resize image
def resize_image(image, target_size):
    img_ratio = image.width / image.height
    target_width, target_height = target_size
    if img_ratio > 1:
        new_width = target_width
        new_height = int(target_width / img_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * img_ratio)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

# Real-time recognition and keyword detection function
def real_time_recognition():
    transcription_buffer = []  # Buffer to collect transcriptions for batching

    with sd.RawInputStream(samplerate=samplerate, blocksize=int(samplerate * duration), dtype='int16',
                           channels=1, callback=callback):
        while True:
            audio_chunk = audio_queue.get()
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe audio chunk
            segments, info = model.transcribe(audio_data, beam_size=5, language='en')
            chunk_transcription = " ".join([segment.text for segment in segments])
            transcription_buffer.append(chunk_transcription)

            # Process transcription buffer when it reaches a threshold (e.g., 3 chunks)
            if len(transcription_buffer) >= 3:
                combined_text = " ".join(transcription_buffer)
                gui_queue.put({"text": combined_text, "type": "transcription"})

                # Extract keywords from the combined transcription
                extracted_keywords = extract_keywords(combined_text)

                # Match keywords with JSON entries (Case-Sensitive)
                matched_images = []
                for keyword in extracted_keywords:
                    # Check if any keyword matches exactly or as a substring in JSON keys (case-sensitive)
                    for json_key in keyword_image_map.keys():
                        if json_key in keyword or keyword in json_key:
                            matched_images.append(keyword_image_map[json_key])
                            gui_queue.put({"image": keyword_image_map[json_key], "type": "image"})

                # Log matched keywords and images
                gui_queue.put({"keywords": extracted_keywords, "type": "keywords"})

                # Clear the buffer for the next batch
                transcription_buffer = []

# Streamlit app layout and functionality
def main():
    st.title("🎓 Interactive Learning Assistant")
    
    # Header image
    st.image("https://example.com/path/to/header_image.jpg", use_column_width=True)
    
    # Instructions in an expander
    with st.expander("📚 Instructions", expanded=False):
        st.markdown("""
        1. Click the Start Recording button to begin.
        2. Speak clearly into your microphone.
        3. Watch as the app transcribes your speech and extracts keywords.
        4. Related images will be displayed based on the keywords detected.
        5. Click Stop Recording when you're done.
        """)
    
    # Configuration in an expander
    with st.expander("⚙ Configuration", expanded=False):
        st.file_uploader("Upload custom keyword-image mapping (JSON)", type="json")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Speech Recognition and Keyword Extraction")
        start_button = st.button("🎙 Start Recording")
        stop_button = st.button("🛑 Stop Recording")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        transcription_area = st.empty()
        keywords_area = st.empty()
    
    with col2:
        st.subheader("Related Image")
        image_area = st.empty()
    
    # Recording state
    recording = False
    
    if start_button:
        recording = True
        threading.Thread(target=real_time_recognition, daemon=True).start()
        status_text.success("Recording started! Speak now.")
    
    if stop_button:
        recording = False
        status_text.info("Recording stopped.")
    
    # Continuously update the UI
    while recording:
        if not gui_queue.empty():
            data = gui_queue.get()
            if data["type"] == "transcription":
                transcription_area.markdown(f"Transcribed Text:\n{data['text']}", unsafe_allow_html=True)
            elif data["type"] == "keywords":
                keywords_area.markdown(f"Extracted Keywords:\n{', '.join(data['keywords'])}", unsafe_allow_html=True)
            elif data["type"] == "image":
                image_path = data["image"]
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    img_resized = resize_image(img, (300, 300))
                    image_area.image(img_resized, caption="Keyword Image", use_column_width=True)
                else:
                    image_area.error(f"Image not found: {image_path}")

        # Update progress bar for visual feedback
        progress = (time.time() % 5) / 5  # Cycles every 5 seconds
        progress_bar.progress(progress)
        time.sleep(0.1)

if __name__ == "__main__":
    main()
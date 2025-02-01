import os
import json
import streamlit as st
import sounddevice as sd
import numpy as np
import queue
import threading
import time
from PIL import Image
from faster_whisper import WhisperModel
import spacy

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
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Create queues
audio_queue = queue.Queue()
gui_queue = queue.Queue()

# Define audio stream parameters
samplerate = 16000
duration = 5

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
    with sd.RawInputStream(samplerate=samplerate, blocksize=int(samplerate * duration), dtype='int16',
                           channels=1, callback=callback):
        while True:
            audio_chunk = audio_queue.get()
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

            segments, info = model.transcribe(audio_data, beam_size=5, language='en')
            for segment in segments:
                text = segment.text
                gui_queue.put({"text": text, "type": "transcription"})
                keywords = extract_keywords(text)
                gui_queue.put({"keywords": keywords, "type": "keywords"})

                for keyword in keywords:
                    if keyword in keyword_image_map:
                        gui_queue.put({"image": keyword_image_map[keyword], "type": "image"})

# Directory to store uploaded images
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Streamlit App
st.title("🎓 Interactive Learning Assistant")

# Upload custom keyword-image mapping (JSON)
uploaded_json = st.file_uploader("Upload custom keyword-image mapping (JSON)", type="json")
if uploaded_json:
    try:
        keyword_image_map = json.load(uploaded_json)
        st.success("Custom keyword-image mapping loaded successfully.")
    except json.JSONDecodeError:
        st.error("Failed to load the JSON file. Please ensure it is in the correct format.")

# Initialize session state for dataset
if "dataset" not in st.session_state:
    st.session_state.dataset = {}

# Image upload and dataset creation
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    file_path = save_uploaded_file(uploaded_file)
    st.success(f"Uploaded {uploaded_file.name}")

    # Input Keywords
    keywords = st.text_input("Enter keywords for this image (comma-separated):", placeholder="e.g., peas, hill, turing")

    if st.button("Add to Dataset"):
        if keywords.strip():
            for kw in keywords.split(","):
                keyword = kw.strip()
                st.session_state.dataset[keyword] = uploaded_file.name
            st.success(f"Keywords for image '{uploaded_file.name}' added.")
        else:
            st.error("Please enter at least one keyword.")

# Save Dataset to JSON
if st.button("Save Dataset to JSON"):
    if st.session_state.dataset:
        output_file = "image_dataset.json"
        with open(output_file, "w") as f:
            json.dump(st.session_state.dataset, f, indent=4)
        st.success(f"Dataset saved to {output_file}")
    else:
        st.warning("No data to save. Please add images and keywords first.")

# Start and stop buttons for audio recording
start_button = st.button("🎙 Start Recording")
stop_button = st.button("🛑 Stop Recording")

# Initialize progress bar and status
progress_bar = st.progress(0)
status_text = st.empty()

# Initialize areas for displaying transcription and keywords
transcription_area = st.empty()
keywords_area = st.empty()
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

# Continuously update the UI during recording
while recording:
    if not gui_queue.empty():
        data = gui_queue.get()
        if data["type"] == "transcription":
            transcription_area.markdown(f"*Transcribed Text:*\n{data['text']}", unsafe_allow_html=True)
        elif data["type"] == "keywords":
            keywords_area.markdown(f"*Extracted Keywords:*\n{', '.join(data['keywords'])}", unsafe_allow_html=True)
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


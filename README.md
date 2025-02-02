# 🎓 real-time-audio-visualizer
📌 Real-Time Audio-Driven Image Display An interactive learning tool that converts real-time speech into text and dynamically displays relevant images. Powered by Whisper AI for speech recognition, spaCy for keyword extraction, and Streamlit for a seamless UI. Ideal for education, presentations, and live demonstrations. 

### 📷Project Screenshots

## 🚀 Features
- 🎤 **Real-time Speech Recognition** using Whisper AI
- 🔍 **Keyword Extraction** with spaCy NLP
- 🖼️ **Automatic Image Display** based on extracted keywords
- 📁 **Custom Dataset Support** for personalized learning
- 🎛 **Interactive UI** with real-time updates

---

## 🛠️ Tech Stack
| Technology       | Usage |
|-----------------|-----------------------------------|
| **Python**      | Core programming language |
| **Streamlit**   | Interactive Web UI |
| **Whisper AI**  | Speech-to-Text Transcription |
| **spaCy**       | NLP-based Keyword Extraction |
| **Sounddevice** | Real-time Audio Capture |
| **Pillow**      | Image Processing |

---

## 🏗️ Architecture & Workflow
1. 🎙️ **User speaks** into the microphone.
2. 📝 **Whisper AI transcribes** the speech into text.
3. 🔎 **spaCy extracts** key concepts (nouns, proper nouns).
4. 🖼 **Matching images** are displayed based on keywords.
5. 📊 **Real-time UI updates** enable seamless interaction.

---

## 🛠 Installation & Setup

### 📌 Prerequisites
- Python 3.8+
- Virtual Environment (Recommended)

### 🔧 Installation Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Tanishka-Singh05/real-time-audio-visualizer.git
   cd interactive-learning-assistant
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
4. **Run the Application**
   ```bash
   streamlit run app.py

### 🤝 Contributing
Want to improve this project? Feel free to fork, submit issues, or create pull requests!


#from faster_whisper import WhisperModel
#model = WhisperModel("small", device="cuda", compute_type="float16")
#print("Model loaded successfully")

import sounddevice as sd
import numpy as np

samplerate = 16000
duration = 5  # seconds
print("Recording...")
audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
sd.wait()
print("Recording complete")


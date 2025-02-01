from faster_whisper import WhisperModel

model_size = "large-v3"

# Run on GPU with float16
model = WhisperModel("small", device="cpu", compute_type="int8")



segments, info = model.transcribe("How AI Could Save (Not Destroy) Education  Sal Khan  TED.mp3", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
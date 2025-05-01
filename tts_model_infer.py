from TTS.api import TTS

print("Loaded TTS")
# print(TTS.list_models())
tts = TTS("tts_models/multi-dataset/xtts_v2", gpu=False)
# tts.model
import time

start = time.time()
tts.tts_to_file(
    text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
    file_path="output.wav",
    # speaker="Brenda Stern",
    speaker_wav=[
        "./geralt/0x00112c9e.wav",
        "./geralt/0x001133d2.wav",
        "./geralt/0x001136dd.wav",
        "./geralt/0x0011351f.wav",
        "./geralt/0x0011fc7c.wav",
    ],
    language="en",
    split_sentences=True,
)

print(time.time() - start)

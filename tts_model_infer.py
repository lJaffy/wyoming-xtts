from TTS.api import TTS
from torch.cuda import is_available
from pathlib import Path

# from Trainer
model_name = "tts_models/en/ljspeech/glow-tts"
model_name = "tts_models/en/ljspeech/tacotron2-DDC"
model_name = "tts_models/en/vctk/vits"
model_name = "tts_models/multilingual/multi-dataset/your_tts"
model_name = "tts_models/multi-dataset/xtts_v2"

print("Loaded TTS")
print(TTS.list_models())
# tts = TTS("tts_models/multi-dataset/xtts_v2", gpu=is_available())
# tts = TTS("tts_models/en/ljspeech/tacotron2-DCA", gpu=is_available())
tts = TTS(model_name, gpu=is_available())
# tts.model
import time

first_speaker = tts.speakers[0] if tts.speakers else None
# exit()
start = time.time()
tts.tts_to_file(
    text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
    file_path="output.wav",
    # speaker=first_speaker,
    speaker_wav=list(Path("custom/geralt").glob("*.wav")),
    language="en",
    split_sentences=True,
)

print(time.time() - start)

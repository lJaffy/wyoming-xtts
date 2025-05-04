import time
import torch
import torchaudio
from TTS.tts.models.xtts import Xtts
import TTS.api
from scipy.io.wavfile import write
from TTS.config import load_config

manager = TTS.api.ModelManager()

model_path, config_path, model = manager.download_model(
    "tts_models/multi-dataset/xtts_v2"
)
print("Loading model...")
config = load_config(config_path)
model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_dir=model_path,
    use_deepspeed=False,
)
# model.cuda()
print(model.speaker_manager.speaker_names)
print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=[
        "./geralt/0x00112c9e.wav",
        "./geralt/0x001133d2.wav",
        "./geralt/0x001136dd.wav",
        "./geralt/0x0011351f.wav",
        "./geralt/0x0011fc7c.wav",
    ],
    sound_norm_refs=True,
)
stream = True
print(speaker_embedding)
print("Inference...")
t0 = time.time()
if stream:
    chunks = model.inference_stream(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "en",
        gpt_cond_latent,
        speaker_embedding,
        enable_text_splitting=False,
    )

    wav_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            print(f"Time to first chunck: {time.time() - t0}")
        print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
        from io import BytesIO

        bytestring = BytesIO()
        chunk_fn = f"audio_stream/chunk_{i}.wav"
        torchaudio.save(
            chunk_fn, chunk.squeeze().unsqueeze(0).cpu(), 24000, format="wav"
        )
        #        playsound.playsound(chunk_fn, block=False)
        wav_chunks.append(chunk)
    wav = torch.cat(wav_chunks, dim=0)
    torchaudio.save("xtts_streaming.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)
else:
    wav = model.inference(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "en",
        gpt_cond_latent,
        speaker_embedding,
    )["wav"]
    write("xtts_infer.wav", 24000, wav)
print(f"Time to last chunck: {time.time() - t0}")

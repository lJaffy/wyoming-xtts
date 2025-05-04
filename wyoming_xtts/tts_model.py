import wave
from typing import Iterable
import torch
from TTS.tts.models.xtts import Xtts
import TTS.api
from TTS.config import load_config
from pathlib import Path
from typing import Any
from pydantic import BaseModel
import numpy as np


def load_model_from_name(model_name: str, use_cuda: bool = False) -> Xtts:
    manager = TTS.api.ModelManager()

    model_path, config_path, model = manager.download_model(model_name)
    print("Loading model...")
    config = load_config(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=model_path,
        use_deepspeed=False,
    )

    if use_cuda:
        model.cuda()
    print("Model loaded")
    return model


class CustomSpeaker(BaseModel):
    name: str
    gpt_cond_latent: Any
    speaker_embedding: Any
    source_data: list[Path]
    language: str

    class Config:
        arbitary_types_allowed = True


def generate_custom_speaker(
    model: Xtts, name: str, audio_samples=list[Path]
) -> CustomSpeaker:
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=audio_samples,
        sound_norm_refs=True,
    )
    return CustomSpeaker(
        name=name,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        source_data=audio_samples,
        language="en",
    )


def register_custom_speaker(model: Xtts, speaker: CustomSpeaker) -> None:
    model.speaker_manager.speakers[speaker.name] = {
        "gpt_cond_latent": speaker.gpt_cond_latent,
        "speaker_embedding": speaker.speaker_embedding,
    }


def stream_text(model: Xtts, text: str, speaker_name: str):
    chunks: Iterable[torch.Tensor] = model.inference_stream(
        text,
        enable_text_splitting=False,
        gpt_cond_latent=model.speaker_manager.speakers[speaker_name]["gpt_cond_latent"],
        speaker_embedding=model.speaker_manager.speakers[speaker_name][
            "speaker_embedding"
        ],
        language="en",
    )

    for i, chunk in enumerate(chunks):
        print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
        chunk_dat = chunk.squeeze().unsqueeze(0).cpu().numpy()

        yield (chunk_dat * 32767).astype(np.int16).tobytes()


def main():
    model = load_model_from_name("tts_models/multi-dataset/xtts_v2")
    geralt = generate_custom_speaker(model, "Geralt", [])
    register_custom_speaker(model, geralt)
    input_string = "Well well well, I was going to come in here looking for a fight, but it looks like somebody beat me to it! You look like shit."

    fh = wave.open("Reconstructed.wav", "wb")
    fh.setsampwidth(2)
    fh.setframerate(24000)
    fh.setnchannels(1)
    for wav_chunk in stream_text(model, input_string, geralt):
        print("Added a chunk!")
        fh.writeframes(wav_chunk)
    fh.close()

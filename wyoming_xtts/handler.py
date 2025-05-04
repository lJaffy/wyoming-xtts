"""Event handler for clients of the server."""

import argparse
import json
import logging
import math
import os
import wave
from typing import Any, Dict, Optional

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize, SynthesizeVoice

from .process import xTTSProcessManager
from .tts_model import (
    load_model_from_name,
    register_custom_speaker,
    generate_custom_speaker,
    stream_text,
    CustomSpeaker,
    Xtts,
)

_LOGGER = logging.getLogger(__name__)

SAMPLE_WIDTH = 2
FRAME_RATE = 24_000
CHANNELS = 1


class xTTSEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        process_manager: xTTSProcessManager,
        model: Xtts,
        speaker: CustomSpeaker,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.speaker = speaker
        self.process_manager = process_manager

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        if not Synthesize.is_type(event.type):
            _LOGGER.warning("Unexpected event: %s", event)
            return True

        try:
            return await self._handle_event(event)
        except Exception as err:
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            raise err

    async def _handle_event(self, event: Event) -> bool:
        synthesize = Synthesize.from_event(event)
        _LOGGER.debug(synthesize)

        raw_text = synthesize.text
        if synthesize.voice is None:
            synthesize.voice = SynthesizeVoice(name="Geralt")
            print(synthesize.voice)
        # Join multiple lines
        text = " ".join(raw_text.strip().splitlines())

        rate = FRAME_RATE
        width = SAMPLE_WIDTH
        channels = CHANNELS

        await self.write_event(
            AudioStart(
                rate=rate,
                width=width,
                channels=channels,
            ).event(),
        )

        for chunk in stream_text(self.model, text, synthesize.voice.name):
            await self.write_event(
                AudioChunk(
                    audio=chunk,
                    rate=rate,
                    width=width,
                    channels=channels,
                ).event(),
            )

        await self.write_event(AudioStop().event())
        _LOGGER.debug("Completed request")

        return True

#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, Set

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice, TtsVoiceSpeaker
from wyoming.server import AsyncServer

from . import __version__
from .download import find_voice, get_voices
from .handler import xTTSEventHandler
from .process import xTTSProcessManager
from .tts_model import (
    load_model_from_name,
    generate_custom_speaker,
    register_custom_speaker,
)

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="stdio://", help="unix:// or tcp://")
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--speaker", type=str, help="Name or id of speaker for default voice"
    )
    parser.add_argument(
        "--max-procs",
        type=int,
        default=1,
        help="Maximum number of TTS process to run simultaneously (default: 1)",
    )
    #
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)
    model = load_model_from_name("tts_models/multi-dataset/xtts_v2")
    speaker = generate_custom_speaker(model, "Geralt", [])
    register_custom_speaker(model, speaker)
    voices = []
    print("Loading Voices")
    for speaker in model.speaker_manager.speaker_names:
        print(f"Speaker: {speaker}")
        print(model.speaker_manager.speakers[speaker].keys())
        voices.append(
            TtsVoice(
                name=speaker,
                description=speaker,
                version=None,
                attribution=Attribution(name="", url=""),
                installed=True,
                languages=["en"],
            )
        )

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="xTTS",
                description="A fast, local, neural text to speech engine",
                attribution=Attribution(
                    name="rhasspy", url="https://github.com/rhasspy/piper"
                ),
                installed=True,
                voices=sorted(voices, key=lambda v: v.name),
                version=__version__,
            )
        ],
    )

    process_manager = xTTSProcessManager(args, {})

    # Make sure default voice is loaded.
    # Other voices will be loaded on-demand.
    #    await process_manager.get_process()

    # Start server
    server = AsyncServer.from_uri(args.uri)

    _LOGGER.info("Ready")
    await server.run(
        partial(
            xTTSEventHandler,
            wyoming_info,
            args,
            process_manager,
            model,
            speaker,
        )
    )


# -----------------------------------------------------------------------------


def get_description(voice_info: Dict[str, Any]):
    """Get a human readable description for a voice."""
    name = voice_info["name"]
    name = " ".join(name.split("_"))
    quality = voice_info["quality"]

    return f"{name} ({quality})"


# -----------------------------------------------------------------------------


def run():
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass

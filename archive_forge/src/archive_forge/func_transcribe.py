from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, cast
from argparse import ArgumentParser
from .._utils import get_client, print_model
from ..._types import NOT_GIVEN
from .._models import BaseModel
from .._progress import BufferReader
@staticmethod
def transcribe(args: CLITranscribeArgs) -> None:
    with open(args.file, 'rb') as file_reader:
        buffer_reader = BufferReader(file_reader.read(), desc='Upload progress')
    model = get_client().audio.transcriptions.create(file=(args.file, buffer_reader), model=args.model, language=args.language or NOT_GIVEN, temperature=args.temperature or NOT_GIVEN, prompt=args.prompt or NOT_GIVEN, response_format=cast(Any, args.response_format))
    print_model(model)
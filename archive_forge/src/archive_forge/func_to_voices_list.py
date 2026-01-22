from __future__ import annotations
import uuid
from typing import (
import param
from panel.widgets import Widget
from ..models.text_to_speech import TextToSpeech as _BkTextToSpeech
@staticmethod
def to_voices_list(voices):
    """Returns a list of Voice objects from the list of dicts provided"""
    result = []
    for _voice in voices:
        voice = Voice(**_voice)
        result.append(voice)
    return result
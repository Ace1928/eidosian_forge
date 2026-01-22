from __future__ import annotations
import uuid
from typing import (
import param
from panel.widgets import Widget
from ..models.text_to_speech import TextToSpeech as _BkTextToSpeech
@staticmethod
def group_by_lang(voices):
    """Returns a dictionary where the key is the `lang` and the value is a list of voices
        for that language."""
    if not voices:
        return {}
    sorted_lang = sorted(list(set((voice.lang for voice in voices))))
    result = {lang: [] for lang in sorted_lang}
    for voice in voices:
        result[voice.lang].append(voice)
    result = {key: sorted(value, key=lambda x: x.name) for key, value in result.items()}
    return result
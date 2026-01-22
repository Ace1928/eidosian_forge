from __future__ import annotations
from typing import (
import param
from ..models.speech_to_text import SpeechToText as _BkSpeechToText
from .base import Widget
from .button import BUTTON_TYPES
@property
def results_as_html(self) -> str:
    """
        Returns the `results` formatted as html

        Convenience method for ease of use
        """
    if not self.results:
        return 'No results'
    html = '<div class="pn-speech-recognition-result">'
    total = len(self.results) - 1
    for index, result in enumerate(reversed(self.results_deserialized)):
        if len(self.results) > 1:
            html += f'<h3>Result {total - index}</h3>'
        html += f'<span>Is Final: {result.is_final}</span><br/>'
        for index2, alternative in enumerate(result.alternatives):
            if len(result.alternatives) > 1:
                html += f'<h4>Alternative {index2}</h4>'
            html += f'\n                <span>Confidence: {alternative.confidence:.2f}</span>\n                </br>\n                <p>\n                  <strong>{alternative.transcript}</strong>\n                </p>\n                '
    html += '</div>'
    return html
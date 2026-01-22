from __future__ import annotations
import json
from typing import Dict, Literal, TypedDict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from typing_extensions import NotRequired
from speech_recognition.audio import AudioData
from speech_recognition.exceptions import RequestError, UnknownValueError
@staticmethod
def find_best_hypothesis(alternatives: list[Alternative]) -> Alternative:
    """
        >>> alternatives = [{"transcript": "one two three", "confidence": 0.42899391}, {"transcript": "1 2", "confidence": 0.49585345}]
        >>> OutputParser.find_best_hypothesis(alternatives)
        {'transcript': 'one two three', 'confidence': 0.42899391}

        >>> alternatives = [{"confidence": 0.49585345}]
        >>> OutputParser.find_best_hypothesis(alternatives)
        Traceback (most recent call last):
          ...
        speech_recognition.exceptions.UnknownValueError
        """
    if 'confidence' in alternatives:
        best_hypothesis: Alternative = max(alternatives, key=lambda alternative: alternative['confidence'])
    else:
        best_hypothesis: Alternative = alternatives[0]
    if 'transcript' not in best_hypothesis:
        raise UnknownValueError()
    return best_hypothesis
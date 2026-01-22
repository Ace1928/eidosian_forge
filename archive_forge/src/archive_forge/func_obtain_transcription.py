from __future__ import annotations
import json
from typing import Dict, Literal, TypedDict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from typing_extensions import NotRequired
from speech_recognition.audio import AudioData
from speech_recognition.exceptions import RequestError, UnknownValueError
def obtain_transcription(request: Request, timeout: int) -> str:
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as e:
        raise RequestError('recognition request failed: {}'.format(e.reason))
    except URLError as e:
        raise RequestError('recognition connection failed: {}'.format(e.reason))
    return response.read().decode('utf-8')
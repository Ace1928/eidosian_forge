import io
from typing import Any, Iterable, List, Optional
from urllib.parse import urlencode
from multidict import MultiDict, MultiDictProxy
from . import hdrs, multipart, payload
from .helpers import guess_filename
from .payload import Payload
@property
def is_multipart(self) -> bool:
    return self._is_multipart
import asyncio
import collections.abc
import datetime
import enum
import json
import math
import time
import warnings
from concurrent.futures import Executor
from http import HTTPStatus
from http.cookies import SimpleCookie
from typing import (
from multidict import CIMultiDict, istr
from . import hdrs, payload
from .abc import AbstractStreamWriter
from .compression_utils import ZLibCompressor
from .helpers import (
from .http import SERVER_SOFTWARE, HttpVersion10, HttpVersion11
from .payload import Payload
from .typedefs import JSONEncoder, LooseHeaders
def enable_chunked_encoding(self, chunk_size: Optional[int]=None) -> None:
    """Enables automatic chunked transfer encoding."""
    self._chunked = True
    if hdrs.CONTENT_LENGTH in self._headers:
        raise RuntimeError("You can't enable chunked encoding when a content length is set")
    if chunk_size is not None:
        warnings.warn('Chunk size is deprecated #1615', DeprecationWarning)
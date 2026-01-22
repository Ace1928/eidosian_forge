import abc
import asyncio
import re
import string
from contextlib import suppress
from enum import IntEnum
from typing import (
from multidict import CIMultiDict, CIMultiDictProxy, istr
from yarl import URL
from . import hdrs
from .base_protocol import BaseProtocol
from .compression_utils import HAS_BROTLI, BrotliDecompressor, ZLibDecompressor
from .helpers import (
from .http_exceptions import (
from .http_writer import HttpVersion, HttpVersion10
from .log import internal_logger
from .streams import EMPTY_PAYLOAD, StreamReader
from .typedefs import RawHeaders
def get_content_length() -> Optional[int]:
    length_hdr = msg.headers.get(CONTENT_LENGTH)
    if length_hdr is None:
        return None
    if not DIGITS.fullmatch(length_hdr):
        raise InvalidHeader(CONTENT_LENGTH)
    return int(length_hdr)
import base64
import binascii
import json
import re
import uuid
import warnings
import zlib
from collections import deque
from types import TracebackType
from typing import (
from urllib.parse import parse_qsl, unquote, urlencode
from multidict import CIMultiDict, CIMultiDictProxy, MultiMapping
from .compression_utils import ZLibCompressor, ZLibDecompressor
from .hdrs import (
from .helpers import CHAR, TOKEN, parse_mimetype, reify
from .http import HeadersParser
from .payload import (
from .streams import StreamReader
def enable_encoding(self, encoding: str) -> None:
    if encoding == 'base64':
        self._encoding = encoding
        self._encoding_buffer = bytearray()
    elif encoding == 'quoted-printable':
        self._encoding = 'quoted-printable'
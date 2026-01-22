import asyncio
import codecs
import contextlib
import functools
import io
import re
import sys
import traceback
import warnings
from hashlib import md5, sha1, sha256
from http.cookies import CookieError, Morsel, SimpleCookie
from types import MappingProxyType, TracebackType
from typing import (
import attr
from multidict import CIMultiDict, CIMultiDictProxy, MultiDict, MultiDictProxy
from yarl import URL
from . import hdrs, helpers, http, multipart, payload
from .abc import AbstractStreamWriter
from .client_exceptions import (
from .compression_utils import HAS_BROTLI
from .formdata import FormData
from .helpers import (
from .http import (
from .log import client_logger
from .streams import StreamReader
from .typedefs import (
def update_transfer_encoding(self) -> None:
    """Analyze transfer-encoding header."""
    te = self.headers.get(hdrs.TRANSFER_ENCODING, '').lower()
    if 'chunked' in te:
        if self.chunked:
            raise ValueError('chunked can not be set if "Transfer-Encoding: chunked" header is set')
    elif self.chunked:
        if hdrs.CONTENT_LENGTH in self.headers:
            raise ValueError('chunked can not be set if Content-Length header is set')
        self.headers[hdrs.TRANSFER_ENCODING] = 'chunked'
    elif hdrs.CONTENT_LENGTH not in self.headers:
        self.headers[hdrs.CONTENT_LENGTH] = str(len(self.body))
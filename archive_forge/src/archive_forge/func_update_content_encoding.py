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
def update_content_encoding(self, data: Any) -> None:
    """Set request content encoding."""
    if data is None:
        return
    enc = self.headers.get(hdrs.CONTENT_ENCODING, '').lower()
    if enc:
        if self.compress:
            raise ValueError('compress can not be set if Content-Encoding header is set')
    elif self.compress:
        if not isinstance(self.compress, str):
            self.compress = 'deflate'
        self.headers[hdrs.CONTENT_ENCODING] = self.compress
        self.chunked = True
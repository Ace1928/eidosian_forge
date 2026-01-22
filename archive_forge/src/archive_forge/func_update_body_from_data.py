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
def update_body_from_data(self, body: Any) -> None:
    if body is None:
        return
    if isinstance(body, FormData):
        body = body()
    try:
        body = payload.PAYLOAD_REGISTRY.get(body, disposition=None)
    except payload.LookupError:
        body = FormData(body)()
    self.body = body
    if not self.chunked:
        if hdrs.CONTENT_LENGTH not in self.headers:
            size = body.size
            if size is None:
                self.chunked = True
            elif hdrs.CONTENT_LENGTH not in self.headers:
                self.headers[hdrs.CONTENT_LENGTH] = str(size)
    assert body.headers
    for key, value in body.headers.items():
        if key in self.headers:
            continue
        if key in self.skip_auto_headers:
            continue
        self.headers[key] = value
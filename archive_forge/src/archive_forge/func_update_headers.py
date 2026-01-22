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
def update_headers(self, headers: Optional[LooseHeaders]) -> None:
    """Update request headers."""
    self.headers: CIMultiDict[str] = CIMultiDict()
    netloc = cast(str, self.url.raw_host)
    if helpers.is_ipv6_address(netloc):
        netloc = f'[{netloc}]'
    netloc = netloc.rstrip('.')
    if self.url.port is not None and (not self.url.is_default_port()):
        netloc += ':' + str(self.url.port)
    self.headers[hdrs.HOST] = netloc
    if headers:
        if isinstance(headers, (dict, MultiDictProxy, MultiDict)):
            headers = headers.items()
        for key, value in headers:
            if key.lower() == 'host':
                self.headers[key] = value
            else:
                self.headers.add(key, value)
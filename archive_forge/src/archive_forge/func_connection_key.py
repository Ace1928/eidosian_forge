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
@property
def connection_key(self) -> ConnectionKey:
    proxy_headers = self.proxy_headers
    if proxy_headers:
        h: Optional[int] = hash(tuple(((k, v) for k, v in proxy_headers.items())))
    else:
        h = None
    return ConnectionKey(self.host, self.port, self.is_ssl(), self.ssl, self.proxy, self.proxy_auth, h)
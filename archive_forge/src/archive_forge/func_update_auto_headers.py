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
def update_auto_headers(self, skip_auto_headers: Iterable[str]) -> None:
    self.skip_auto_headers = CIMultiDict(((hdr, None) for hdr in sorted(skip_auto_headers)))
    used_headers = self.headers.copy()
    used_headers.extend(self.skip_auto_headers)
    for hdr, val in self.DEFAULT_HEADERS.items():
        if hdr not in used_headers:
            self.headers.add(hdr, val)
    if hdrs.USER_AGENT not in used_headers:
        self.headers[hdrs.USER_AGENT] = SERVER_SOFTWARE
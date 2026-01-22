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
def update_version(self, version: Union[http.HttpVersion, str]) -> None:
    """Convert request version to two elements tuple.

        parser HTTP version '1.1' => (1, 1)
        """
    if isinstance(version, str):
        v = [part.strip() for part in version.split('.', 1)]
        try:
            version = http.HttpVersion(int(v[0]), int(v[1]))
        except ValueError:
            raise ValueError(f'Can not parse http version number: {version}') from None
    self.version = version
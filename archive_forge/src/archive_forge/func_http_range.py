import asyncio
import datetime
import io
import re
import socket
import string
import tempfile
import types
import warnings
from http.cookies import SimpleCookie
from types import MappingProxyType
from typing import (
from urllib.parse import parse_qsl
import attr
from multidict import (
from yarl import URL
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import (
from .http_parser import RawRequestMessage
from .http_writer import HttpVersion
from .multipart import BodyPartReader, MultipartReader
from .streams import EmptyStreamReader, StreamReader
from .typedefs import (
from .web_exceptions import HTTPRequestEntityTooLarge
from .web_response import StreamResponse
@reify
def http_range(self) -> slice:
    """The content of Range HTTP header.

        Return a slice instance.

        """
    rng = self._headers.get(hdrs.RANGE)
    start, end = (None, None)
    if rng is not None:
        try:
            pattern = '^bytes=(\\d*)-(\\d*)$'
            start, end = re.findall(pattern, rng)[0]
        except IndexError:
            raise ValueError('range not in acceptable format')
        end = int(end) if end else None
        start = int(start) if start else None
        if start is None and end is not None:
            start = -end
            end = None
        if start is not None and end is not None:
            end += 1
            if start >= end:
                raise ValueError('start cannot be after end')
        if start is end is None:
            raise ValueError('No start or end of range specified')
    return slice(start, end, 1)
import asyncio
import collections.abc
import datetime
import enum
import json
import math
import time
import warnings
from concurrent.futures import Executor
from http import HTTPStatus
from http.cookies import SimpleCookie
from typing import (
from multidict import CIMultiDict, istr
from . import hdrs, payload
from .abc import AbstractStreamWriter
from .compression_utils import ZLibCompressor
from .helpers import (
from .http import SERVER_SOFTWARE, HttpVersion10, HttpVersion11
from .payload import Payload
from .typedefs import JSONEncoder, LooseHeaders
def enable_compression(self, force: Optional[Union[bool, ContentCoding]]=None) -> None:
    """Enables response compression encoding."""
    if type(force) == bool:
        force = ContentCoding.deflate if force else ContentCoding.identity
        warnings.warn('Using boolean for force is deprecated #3318', DeprecationWarning)
    elif force is not None:
        assert isinstance(force, ContentCoding), 'force should one of None, bool or ContentEncoding'
    self._compression = True
    self._compression_force = force
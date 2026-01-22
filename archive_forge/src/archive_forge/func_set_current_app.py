import abc
import asyncio
import base64
import hashlib
import inspect
import keyword
import os
import re
import warnings
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import (
from yarl import URL, __version__ as yarl_version  # type: ignore[attr-defined]
from . import hdrs
from .abc import AbstractMatchInfo, AbstractRouter, AbstractView
from .helpers import DEBUG
from .http import HttpVersion11
from .typedefs import Handler, PathLike
from .web_exceptions import (
from .web_fileresponse import FileResponse
from .web_request import Request
from .web_response import Response, StreamResponse
from .web_routedef import AbstractRouteDef
@contextmanager
def set_current_app(self, app: 'Application') -> Generator[None, None, None]:
    if DEBUG:
        if app not in self._apps:
            raise RuntimeError('Expected one of the following apps {!r}, got {!r}'.format(self._apps, app))
    prev = self._current_app
    self._current_app = app
    try:
        yield
    finally:
        self._current_app = prev
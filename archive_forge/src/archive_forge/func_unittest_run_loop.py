import asyncio
import contextlib
import gc
import inspect
import ipaddress
import os
import socket
import sys
import warnings
from abc import ABC, abstractmethod
from types import TracebackType
from typing import (
from unittest import IsolatedAsyncioTestCase, mock
from aiosignal import Signal
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL
import aiohttp
from aiohttp.client import _RequestContextManager, _WSRequestContextManager
from . import ClientSession, hdrs
from .abc import AbstractCookieJar
from .client_reqrep import ClientResponse
from .client_ws import ClientWebSocketResponse
from .helpers import sentinel
from .http import HttpVersion, RawRequestMessage
from .typedefs import StrOrURL
from .web import (
from .web_protocol import _RequestHandler
def unittest_run_loop(func: Any, *args: Any, **kwargs: Any) -> Any:
    """
    A decorator dedicated to use with asynchronous AioHTTPTestCase test methods.

    In 3.8+, this does nothing.
    """
    warnings.warn('Decorator `@unittest_run_loop` is no longer needed in aiohttp 3.8+', DeprecationWarning, stacklevel=2)
    return func
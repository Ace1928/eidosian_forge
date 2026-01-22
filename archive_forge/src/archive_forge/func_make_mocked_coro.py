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
def make_mocked_coro(return_value: Any=sentinel, raise_exception: Any=sentinel) -> Any:
    """Creates a coroutine mock."""

    async def mock_coro(*args: Any, **kwargs: Any) -> Any:
        if raise_exception is not sentinel:
            raise raise_exception
        if not inspect.isawaitable(return_value):
            return return_value
        await return_value
    return mock.Mock(wraps=mock_coro)
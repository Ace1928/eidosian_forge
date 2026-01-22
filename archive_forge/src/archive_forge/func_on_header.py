from __future__ import annotations
import asyncio
import http
import logging
import re
import urllib
from asyncio.events import TimerHandle
from collections import deque
from typing import Any, Callable, Literal, cast
import httptools
from uvicorn._types import (
from uvicorn.config import Config
from uvicorn.logging import TRACE_LOG_LEVEL
from uvicorn.protocols.http.flow_control import (
from uvicorn.protocols.utils import (
from uvicorn.server import ServerState
def on_header(self, name: bytes, value: bytes) -> None:
    name = name.lower()
    if name == b'expect' and value.lower() == b'100-continue':
        self.expect_100_continue = True
    self.headers.append((name, value))
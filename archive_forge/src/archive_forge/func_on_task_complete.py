from __future__ import annotations
import asyncio
import logging
import typing
from typing import Literal
from urllib.parse import unquote
import wsproto
from wsproto import ConnectionType, events
from wsproto.connection import ConnectionState
from wsproto.extensions import Extension, PerMessageDeflate
from wsproto.utilities import LocalProtocolError, RemoteProtocolError
from uvicorn._types import (
from uvicorn.config import Config
from uvicorn.logging import TRACE_LOG_LEVEL
from uvicorn.protocols.utils import (
from uvicorn.server import ServerState
def on_task_complete(self, task: asyncio.Task) -> None:
    self.tasks.discard(task)
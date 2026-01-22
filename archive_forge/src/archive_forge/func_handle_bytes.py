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
def handle_bytes(self, event: events.BytesMessage) -> None:
    self.bytes += event.data
    if event.message_finished:
        self.queue.put_nowait({'type': 'websocket.receive', 'bytes': self.bytes})
        self.bytes = b''
        if not self.read_paused:
            self.read_paused = True
            self.transport.pause_reading()
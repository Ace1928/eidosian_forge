from __future__ import annotations
import asyncio
import http
import logging
from typing import Any, Callable, Literal, cast
from urllib.parse import unquote
import h11
from h11._connection import DEFAULT_MAX_INCOMPLETE_EVENT_SIZE
from uvicorn._types import (
from uvicorn.config import Config
from uvicorn.logging import TRACE_LOG_LEVEL
from uvicorn.protocols.http.flow_control import (
from uvicorn.protocols.utils import (
from uvicorn.server import ServerState
def send_400_response(self, msg: str) -> None:
    reason = STATUS_PHRASES[400]
    headers: list[tuple[bytes, bytes]] = [(b'content-type', b'text/plain; charset=utf-8'), (b'connection', b'close')]
    event = h11.Response(status_code=400, headers=headers, reason=reason)
    output = self.conn.send(event)
    self.transport.write(output)
    output = self.conn.send(event=h11.Data(data=msg.encode('ascii')))
    self.transport.write(output)
    output = self.conn.send(event=h11.EndOfMessage())
    self.transport.write(output)
    self.transport.close()
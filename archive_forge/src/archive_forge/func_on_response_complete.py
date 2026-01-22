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
def on_response_complete(self) -> None:
    self.server_state.total_requests += 1
    if self.transport.is_closing():
        return
    self._unset_keepalive_if_required()
    self.timeout_keep_alive_task = self.loop.call_later(self.timeout_keep_alive, self.timeout_keep_alive_handler)
    self.flow.resume_reading()
    if self.conn.our_state is h11.DONE and self.conn.their_state is h11.DONE:
        self.conn.start_next_cycle()
        self.handle_events()
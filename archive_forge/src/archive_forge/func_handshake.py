from __future__ import annotations
import http
import logging
import os
import selectors
import socket
import ssl
import sys
import threading
from types import TracebackType
from typing import Any, Callable, Optional, Sequence, Type
from websockets.frames import CloseCode
from ..extensions.base import ServerExtensionFactory
from ..extensions.permessage_deflate import enable_server_permessage_deflate
from ..headers import validate_subprotocols
from ..http import USER_AGENT
from ..http11 import Request, Response
from ..protocol import CONNECTING, OPEN, Event
from ..server import ServerProtocol
from ..typing import LoggerLike, Origin, Subprotocol
from .connection import Connection
from .utils import Deadline
def handshake(self, process_request: Optional[Callable[[ServerConnection, Request], Optional[Response]]]=None, process_response: Optional[Callable[[ServerConnection, Request, Response], Optional[Response]]]=None, server_header: Optional[str]=USER_AGENT, timeout: Optional[float]=None) -> None:
    """
        Perform the opening handshake.

        """
    if not self.request_rcvd.wait(timeout):
        self.close_socket()
        self.recv_events_thread.join()
        raise TimeoutError('timed out during handshake')
    if self.request is None:
        self.close_socket()
        self.recv_events_thread.join()
        raise ConnectionError('connection closed during handshake')
    with self.send_context(expected_state=CONNECTING):
        self.response = None
        if process_request is not None:
            try:
                self.response = process_request(self, self.request)
            except Exception as exc:
                self.protocol.handshake_exc = exc
                self.logger.error('opening handshake failed', exc_info=True)
                self.response = self.protocol.reject(http.HTTPStatus.INTERNAL_SERVER_ERROR, 'Failed to open a WebSocket connection.\nSee server log for more information.\n')
        if self.response is None:
            self.response = self.protocol.accept(self.request)
        if server_header is not None:
            self.response.headers['Server'] = server_header
        if process_response is not None:
            try:
                response = process_response(self, self.request, self.response)
            except Exception as exc:
                self.protocol.handshake_exc = exc
                self.logger.error('opening handshake failed', exc_info=True)
                self.response = self.protocol.reject(http.HTTPStatus.INTERNAL_SERVER_ERROR, 'Failed to open a WebSocket connection.\nSee server log for more information.\n')
            else:
                if response is not None:
                    self.response = response
        self.protocol.send_response(self.response)
    if self.protocol.state is not OPEN:
        self.recv_events_thread.join(self.close_timeout)
        self.close_socket()
        self.recv_events_thread.join()
    if self.protocol.handshake_exc is not None:
        raise self.protocol.handshake_exc
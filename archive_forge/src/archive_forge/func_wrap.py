from __future__ import annotations
import asyncio
import email.utils
import functools
import http
import inspect
import logging
import socket
import warnings
from types import TracebackType
from typing import (
from ..datastructures import Headers, HeadersLike, MultipleValuesError
from ..exceptions import (
from ..extensions import Extension, ServerExtensionFactory
from ..extensions.permessage_deflate import enable_server_permessage_deflate
from ..headers import (
from ..http import USER_AGENT
from ..protocol import State
from ..typing import ExtensionHeader, LoggerLike, Origin, StatusLike, Subprotocol
from .compatibility import asyncio_timeout
from .handshake import build_response, check_request
from .http import read_request
from .protocol import WebSocketCommonProtocol
def wrap(self, server: asyncio.base_events.Server) -> None:
    """
        Attach to a given :class:`~asyncio.Server`.

        Since :meth:`~asyncio.loop.create_server` doesn't support injecting a
        custom ``Server`` class, the easiest solution that doesn't rely on
        private :mod:`asyncio` APIs is to:

        - instantiate a :class:`WebSocketServer`
        - give the protocol factory a reference to that instance
        - call :meth:`~asyncio.loop.create_server` with the factory
        - attach the resulting :class:`~asyncio.Server` with this method

        """
    self.server = server
    for sock in server.sockets:
        if sock.family == socket.AF_INET:
            name = '%s:%d' % sock.getsockname()
        elif sock.family == socket.AF_INET6:
            name = '[%s]:%d' % sock.getsockname()[:2]
        elif sock.family == socket.AF_UNIX:
            name = sock.getsockname()
        else:
            name = str(sock.getsockname())
        self.logger.info('server listening on %s', name)
    self.closed_waiter = server.get_loop().create_future()
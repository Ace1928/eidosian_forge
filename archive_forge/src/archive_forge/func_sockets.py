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
@property
def sockets(self) -> Iterable[socket.socket]:
    """
        See :attr:`asyncio.Server.sockets`.

        """
    return self.server.sockets
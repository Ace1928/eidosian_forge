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
def conn_handler(sock: socket.socket, addr: Any) -> None:
    deadline = Deadline(open_timeout)
    try:
        if not unix:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        if ssl_context is not None:
            sock.settimeout(deadline.timeout())
            assert isinstance(sock, ssl.SSLSocket)
            sock.do_handshake()
            sock.settimeout(None)
        protocol_select_subprotocol: Optional[Callable[[ServerProtocol, Sequence[Subprotocol]], Optional[Subprotocol]]] = None
        if select_subprotocol is not None:

            def protocol_select_subprotocol(protocol: ServerProtocol, subprotocols: Sequence[Subprotocol]) -> Optional[Subprotocol]:
                assert select_subprotocol is not None
                assert protocol is connection.protocol
                return select_subprotocol(connection, subprotocols)
        protocol = ServerProtocol(origins=origins, extensions=extensions, subprotocols=subprotocols, select_subprotocol=protocol_select_subprotocol, state=CONNECTING, max_size=max_size, logger=logger)
        assert create_connection is not None
        connection = create_connection(sock, protocol, close_timeout=close_timeout)
        connection.handshake(process_request, process_response, server_header, deadline.timeout())
    except Exception:
        sock.close()
        return
    try:
        handler(connection)
    except Exception:
        protocol.logger.error('connection handler failed', exc_info=True)
        connection.close(CloseCode.INTERNAL_ERROR)
    else:
        connection.close()
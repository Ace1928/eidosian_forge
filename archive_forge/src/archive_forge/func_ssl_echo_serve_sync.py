from __future__ import annotations
import os
import socket as stdlib_socket
import ssl
import sys
import threading
from contextlib import asynccontextmanager, contextmanager, suppress
from functools import partial
from ssl import SSLContext
from typing import (
import pytest
from trio import StapledStream
from trio._tests.pytest_plugin import skip_if_optional_else_raise
from trio.abc import ReceiveStream, SendStream
from trio.testing import (
import trio
from .. import _core, socket as tsocket
from .._abc import Stream
from .._core import BrokenResourceError, ClosedResourceError
from .._core._tests.tutil import slow
from .._highlevel_generic import aclose_forcefully
from .._highlevel_open_tcp_stream import open_tcp_stream
from .._highlevel_socket import SocketListener, SocketStream
from .._ssl import NeedHandshakeError, SSLListener, SSLStream, _is_eof
from .._util import ConflictDetector
from ..testing import (
def ssl_echo_serve_sync(sock: stdlib_socket.socket, *, expect_fail: bool=False) -> None:
    try:
        wrapped = SERVER_CTX.wrap_socket(sock, server_side=True, suppress_ragged_eofs=False)
        with wrapped:
            wrapped.do_handshake()
            while True:
                data = wrapped.recv(4096)
                if not data:
                    with suppress(BrokenPipeError, ssl.SSLZeroReturnError):
                        wrapped.unwrap()
                    return
                wrapped.sendall(data)
    except (ConnectionResetError, ConnectionAbortedError):
        return
    except Exception as exc:
        if expect_fail:
            print('ssl_echo_serve_sync got error as expected:', exc)
        else:
            print('ssl_echo_serve_sync got unexpected error:', exc)
            raise
    else:
        if expect_fail:
            raise RuntimeError('failed to fail?')
    finally:
        sock.close()
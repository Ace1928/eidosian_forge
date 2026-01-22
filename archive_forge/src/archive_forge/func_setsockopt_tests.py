from __future__ import annotations
import errno
import inspect
import os
import socket as stdlib_socket
import sys
import tempfile
from socket import AddressFamily, SocketKind
from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union
import attrs
import pytest
from .. import _core, socket as tsocket
from .._core._tests.tutil import binds_ipv6, creates_ipv6
from .._socket import _NUMERIC_ONLY, SocketType, _SocketType, _try_sync
from ..testing import assert_checkpoints, wait_all_tasks_blocked
def setsockopt_tests(sock: SocketType | SocketStream) -> None:
    """Extract these out, to be reused for SocketStream also."""
    if hasattr(tsocket, 'SO_BINDTODEVICE'):
        sock.setsockopt(tsocket.SOL_SOCKET, tsocket.SO_BINDTODEVICE, None, 0)
    sock.setsockopt(tsocket.IPPROTO_TCP, tsocket.TCP_NODELAY, False)
    with pytest.raises(TypeError, match="invalid value for argument 'value'"):
        sock.setsockopt(tsocket.IPPROTO_TCP, tsocket.TCP_NODELAY, False, 5)
    with pytest.raises(TypeError, match="invalid value for argument 'value'"):
        sock.setsockopt(tsocket.IPPROTO_TCP, tsocket.TCP_NODELAY, None)
from __future__ import annotations
import errno
import os
import socket
import ssl
import stat
import sys
from collections.abc import Awaitable
from ipaddress import IPv6Address, ip_address
from os import PathLike, chmod
from socket import AddressFamily, SocketKind
from typing import Any, Literal, cast, overload
from .. import to_thread
from ..abc import (
from ..streams.stapled import MultiListener
from ..streams.tls import TLSStream
from ._eventloop import get_async_backend
from ._resources import aclose_forcefully
from ._synchronization import Event
from ._tasks import create_task_group, move_on_after
def convert_ipv6_sockaddr(sockaddr: tuple[str, int, int, int] | tuple[str, int]) -> tuple[str, int]:
    """
    Convert a 4-tuple IPv6 socket address to a 2-tuple (address, port) format.

    If the scope ID is nonzero, it is added to the address, separated with ``%``.
    Otherwise the flow id and scope id are simply cut off from the tuple.
    Any other kinds of socket addresses are returned as-is.

    :param sockaddr: the result of :meth:`~socket.socket.getsockname`
    :return: the converted socket address

    """
    if isinstance(sockaddr, tuple) and len(sockaddr) == 4:
        host, port, flowinfo, scope_id = sockaddr
        if scope_id:
            host = host.split('%')[0]
            return (f'{host}%{scope_id}', port)
        else:
            return (host, port)
    else:
        return sockaddr
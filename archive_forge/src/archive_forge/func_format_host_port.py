from __future__ import annotations
import sys
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any
import trio
from trio.socket import SOCK_STREAM, SocketType, getaddrinfo, socket
def format_host_port(host: str | bytes, port: int | str) -> str:
    host = host.decode('ascii') if isinstance(host, bytes) else host
    if ':' in host:
        return f'[{host}]:{port}'
    else:
        return f'{host}:{port}'
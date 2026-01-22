import asyncio
import socket
from contextlib import suppress
from typing import Optional  # noqa
def tcp_nodelay(transport: asyncio.Transport, value: bool) -> None:
    sock = transport.get_extra_info('socket')
    if sock is None:
        return
    if sock.family not in (socket.AF_INET, socket.AF_INET6):
        return
    value = bool(value)
    with suppress(OSError):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, value)
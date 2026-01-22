from __future__ import annotations
import asyncio
import urllib.parse
from uvicorn._types import WWWScope
def get_remote_addr(transport: asyncio.Transport) -> tuple[str, int] | None:
    socket_info = transport.get_extra_info('socket')
    if socket_info is not None:
        try:
            info = socket_info.getpeername()
            return (str(info[0]), int(info[1])) if isinstance(info, tuple) else None
        except OSError:
            return None
    info = transport.get_extra_info('peername')
    if info is not None and isinstance(info, (list, tuple)) and (len(info) == 2):
        return (str(info[0]), int(info[1]))
    return None
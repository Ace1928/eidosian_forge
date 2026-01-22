import asyncio
import functools
import pycares
import socket
import sys
from typing import (
from . import error
def gethostbyname(self, host: str, family: socket.AddressFamily) -> asyncio.Future:
    fut = asyncio.Future(loop=self.loop)
    cb = functools.partial(self._callback, fut)
    self._channel.gethostbyname(host, family, cb)
    return fut
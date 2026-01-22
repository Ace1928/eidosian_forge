from __future__ import annotations
import enum
import logging
import uuid
from typing import Generator, List, Optional, Type, Union
from .exceptions import (
from .extensions import Extension
from .frames import (
from .http11 import Request, Response
from .streams import StreamReader
from .typing import LoggerLike, Origin, Subprotocol
def send_ping(self, data: bytes) -> None:
    """
        Send a `Ping frame`_.

        .. _Ping frame:
            https://datatracker.ietf.org/doc/html/rfc6455#section-5.5.2

        Parameters:
            data: payload containing arbitrary binary data.

        """
    self.send_frame(Frame(OP_PING, data))
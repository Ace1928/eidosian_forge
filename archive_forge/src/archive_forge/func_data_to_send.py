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
def data_to_send(self) -> List[bytes]:
    """
        Obtain data to send to the network.

        Call this method immediately after any of the ``receive_*()``,
        ``send_*()``, or :meth:`fail` methods.

        Write resulting data to the connection.

        The empty bytestring :data:`~websockets.protocol.SEND_EOF` signals
        the end of the data stream. When you receive it, half-close the TCP
        connection.

        Returns:
            List[bytes]: Data to write to the connection.

        """
    writes, self.writes = (self.writes, [])
    return writes
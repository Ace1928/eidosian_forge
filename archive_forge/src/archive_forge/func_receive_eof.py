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
def receive_eof(self) -> None:
    """
        Receive the end of the data stream from the network.

        After calling this method:

        - You must call :meth:`data_to_send` and send this data to the network;
          it will return ``[b""]``, signaling the end of the stream, or ``[]``.
        - You aren't expected to call :meth:`events_received`; it won't return
          any new events.

        Raises:
            EOFError: if :meth:`receive_eof` was called earlier.

        """
    self.reader.feed_eof()
    next(self.parser)
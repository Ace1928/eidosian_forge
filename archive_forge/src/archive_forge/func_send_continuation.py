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
def send_continuation(self, data: bytes, fin: bool) -> None:
    """
        Send a `Continuation frame`_.

        .. _Continuation frame:
            https://datatracker.ietf.org/doc/html/rfc6455#section-5.6

        Parameters:
            data: payload containing the same kind of data
                as the initial frame.
            fin: FIN bit; set it to :obj:`True` if this is the last frame
                of a fragmented message and to :obj:`False` otherwise.

        Raises:
            ProtocolError: if a fragmented message isn't in progress.

        """
    if not self.expect_continuation_frame:
        raise ProtocolError('unexpected continuation frame')
    self.expect_continuation_frame = not fin
    self.send_frame(Frame(OP_CONT, data, fin))
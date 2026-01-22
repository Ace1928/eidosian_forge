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
def send_text(self, data: bytes, fin: bool=True) -> None:
    """
        Send a `Text frame`_.

        .. _Text frame:
            https://datatracker.ietf.org/doc/html/rfc6455#section-5.6

        Parameters:
            data: payload containing text encoded with UTF-8.
            fin: FIN bit; set it to :obj:`False` if this is the first frame of
                a fragmented message.

        Raises:
            ProtocolError: if a fragmented message is in progress.

        """
    if self.expect_continuation_frame:
        raise ProtocolError('expected a continuation frame')
    self.expect_continuation_frame = not fin
    self.send_frame(Frame(OP_TEXT, data, fin))
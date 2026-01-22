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
def recv_frame(self, frame: Frame) -> None:
    """
        Process an incoming frame.

        """
    if frame.opcode is OP_TEXT or frame.opcode is OP_BINARY:
        if self.cur_size is not None:
            raise ProtocolError('expected a continuation frame')
        if frame.fin:
            self.cur_size = None
        else:
            self.cur_size = len(frame.data)
    elif frame.opcode is OP_CONT:
        if self.cur_size is None:
            raise ProtocolError('unexpected continuation frame')
        if frame.fin:
            self.cur_size = None
        else:
            self.cur_size += len(frame.data)
    elif frame.opcode is OP_PING:
        pong_frame = Frame(OP_PONG, frame.data)
        self.send_frame(pong_frame)
    elif frame.opcode is OP_PONG:
        pass
    elif frame.opcode is OP_CLOSE:
        self.close_rcvd = Close.parse(frame.data)
        if self.state is CLOSING:
            assert self.close_sent is not None
            self.close_rcvd_then_sent = False
        if self.cur_size is not None:
            raise ProtocolError('incomplete fragmented message')
        if self.state is OPEN:
            self.send_frame(Frame(OP_CLOSE, frame.data))
            self.close_sent = self.close_rcvd
            self.close_rcvd_then_sent = True
            self.state = CLOSING
        if self.side is SERVER:
            self.send_eof()
        self.parser = self.discard()
        next(self.parser)
    else:
        raise AssertionError(f'unexpected opcode: {frame.opcode:02x}')
    self.events.append(frame)
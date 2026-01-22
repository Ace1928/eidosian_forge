from __future__ import annotations
import contextlib
import logging
import random
import socket
import struct
import threading
import uuid
from types import TracebackType
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Type, Union
from ..exceptions import ConnectionClosed, ConnectionClosedOK, ProtocolError
from ..frames import DATA_OPCODES, BytesLike, CloseCode, Frame, Opcode, prepare_ctrl
from ..http11 import Request, Response
from ..protocol import CLOSED, OPEN, Event, Protocol, State
from ..typing import Data, LoggerLike, Subprotocol
from .messages import Assembler
from .utils import Deadline
def send_data(self) -> None:
    """
        Send outgoing data.

        This method requires holding protocol_mutex.

        Raises:
            OSError: When a socket operations fails.

        """
    assert self.protocol_mutex.locked()
    for data in self.protocol.data_to_send():
        if data:
            if self.close_deadline is not None:
                self.socket.settimeout(self.close_deadline.timeout())
            self.socket.sendall(data)
        else:
            try:
                self.socket.shutdown(socket.SHUT_WR)
            except OSError:
                pass
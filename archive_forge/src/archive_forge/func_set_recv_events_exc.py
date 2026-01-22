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
def set_recv_events_exc(self, exc: Optional[BaseException]) -> None:
    """
        Set recv_events_exc, if not set yet.

        This method requires holding protocol_mutex.

        """
    assert self.protocol_mutex.locked()
    if self.recv_events_exc is None:
        self.recv_events_exc = exc
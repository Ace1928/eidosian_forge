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
def process_event(self, event: Event) -> None:
    """
        Process one incoming event.

        This method is overridden in subclasses to handle the handshake.

        """
    assert isinstance(event, Frame)
    if event.opcode in DATA_OPCODES:
        self.recv_messages.put(event)
    if event.opcode is Opcode.PONG:
        self.acknowledge_pings(bytes(event.data))
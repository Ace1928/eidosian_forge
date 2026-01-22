from __future__ import annotations
import asyncio
import codecs
import collections
import logging
import random
import ssl
import struct
import sys
import time
import uuid
import warnings
from typing import (
from ..datastructures import Headers
from ..exceptions import (
from ..extensions import Extension
from ..frames import (
from ..protocol import State
from ..typing import Data, LoggerLike, Subprotocol
from .compatibility import asyncio_timeout
from .framing import Frame
def connection_closed_exc(self) -> ConnectionClosed:
    exc: ConnectionClosed
    if self.close_rcvd is not None and self.close_rcvd.code in OK_CLOSE_CODES and (self.close_sent is not None) and (self.close_sent.code in OK_CLOSE_CODES):
        exc = ConnectionClosedOK(self.close_rcvd, self.close_sent, self.close_rcvd_then_sent)
    else:
        exc = ConnectionClosedError(self.close_rcvd, self.close_sent, self.close_rcvd_then_sent)
    exc.__cause__ = self.transfer_data_exc
    return exc
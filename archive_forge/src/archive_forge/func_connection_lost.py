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
def connection_lost(self, exc: Optional[Exception]) -> None:
    """
        7.1.4. The WebSocket Connection is Closed.

        """
    self.state = State.CLOSED
    self.logger.debug('= connection is CLOSED')
    self.abort_pings()
    self.connection_lost_waiter.set_result(None)
    if True:
        if self.reader is not None:
            if exc is None:
                self.reader.feed_eof()
            else:
                self.reader.set_exception(exc)
        if not self._paused:
            return
        waiter = self._drain_waiter
        if waiter is None:
            return
        self._drain_waiter = None
        if waiter.done():
            return
        if exc is None:
            waiter.set_result(None)
        else:
            waiter.set_exception(exc)
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
@property
def open(self) -> bool:
    """
        :obj:`True` when the connection is open; :obj:`False` otherwise.

        This attribute may be used to detect disconnections. However, this
        approach is discouraged per the EAFP_ principle. Instead, you should
        handle :exc:`~websockets.exceptions.ConnectionClosed` exceptions.

        .. _EAFP: https://docs.python.org/3/glossary.html#term-eafp

        """
    return self.state is State.OPEN and (not self.transfer_data_task.done())
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
def write_frame_sync(self, fin: bool, opcode: int, data: bytes) -> None:
    frame = Frame(fin, Opcode(opcode), data)
    if self.debug:
        self.logger.debug('> %s', frame)
    frame.write(self.transport.write, mask=self.is_client, extensions=self.extensions)
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
def host(self) -> Optional[str]:
    alternative = 'remote_address' if self.is_client else 'local_address'
    warnings.warn(f'use {alternative}[0] instead of host', DeprecationWarning)
    return self._host
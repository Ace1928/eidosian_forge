from __future__ import annotations
import os
import socket
from collections import defaultdict
from contextlib import contextmanager
from queue import Empty
from kombu.exceptions import ChannelError
from kombu.log import get_logger
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
@cached_property
def lock_value(self):
    return f'{socket.gethostname()}.{os.getpid()}'
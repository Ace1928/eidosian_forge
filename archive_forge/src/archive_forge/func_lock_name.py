from __future__ import annotations
import socket
import uuid
from collections import defaultdict
from contextlib import contextmanager
from queue import Empty
from time import monotonic
from kombu.exceptions import ChannelError
from kombu.log import get_logger
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
@cached_property
def lock_name(self):
    return f'{socket.gethostname()}'
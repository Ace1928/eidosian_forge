from concurrent.futures import Future
from contextlib import contextmanager
import functools
import os
from selectors import EVENT_READ
import socket
from queue import Queue, Full as QueueFull
from threading import Lock, Thread
from typing import Optional
from jeepney import Message, MessageType
from jeepney.bus import get_bus
from jeepney.bus_messages import message_bus
from jeepney.wrappers import ProxyBase, unwrap_msg
from .blocking import (
from .common import (
def reset_interrupt(self):
    """Allow calls to .receive() again after .interrupt()

        To avoid race conditions, you should typically wait for threads to
        respond (e.g. by joining them) between interrupting and resetting.
        """
    while (self.stop_key, EVENT_READ) in self.selector.select(timeout=0):
        os.read(self._stop_r, 1024)
import array
from contextlib import contextmanager
import errno
from itertools import count
import logging
from typing import Optional
from outcome import Value, Error
import trio
from trio.abc import Channel
from jeepney.auth import Authenticator, BEGIN
from jeepney.bus import get_bus
from jeepney.fds import FileDescriptor, fds_buf_size
from jeepney.low_level import Parser, MessageType, Message
from jeepney.wrappers import ProxyBase, unwrap_msg
from jeepney.bus_messages import message_bus
from .common import (
@property
def receive_channel(self):
    return self.queue
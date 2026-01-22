import array
from collections import deque
from errno import ECONNRESET
import functools
from itertools import count
import os
from selectors import DefaultSelector, EVENT_READ
import socket
import time
from typing import Optional
from warnings import warn
from jeepney import Parser, Message, MessageType, HeaderFields
from jeepney.auth import Authenticator, BEGIN
from jeepney.bus import get_bus
from jeepney.fds import FileDescriptor, fds_buf_size
from jeepney.wrappers import ProxyBase, unwrap_msg
from jeepney.routing import Router
from jeepney.bus_messages import message_bus
from .common import MessageFilters, FilterHandle, check_replyable
def send_and_get_reply(self, message, *, timeout=None, unwrap=None):
    """Send a message, wait for the reply and return it

        Filters are applied to other messages received before the reply -
        see :meth:`add_filter`.
        """
    check_replyable(message)
    deadline = timeout_to_deadline(timeout)
    if unwrap is None:
        unwrap = False
    else:
        warn('Passing unwrap= to .send_and_get_reply() is deprecated and will break in a future version of Jeepney.', stacklevel=2)
    serial = next(self.outgoing_serial)
    self.send_message(message, serial=serial)
    while True:
        msg_in = self.receive(timeout=deadline_to_timeout(deadline))
        reply_to = msg_in.header.fields.get(HeaderFields.reply_serial, -1)
        if reply_to == serial:
            if unwrap:
                return unwrap_msg(msg_in)
            return msg_in
        self._router.incoming(msg_in)
        for filter in self._filters.matches(msg_in):
            filter.queue.append(msg_in)
import logging
import socket
from collections import defaultdict
from queue import Queue
from vine import ensure_promise
from . import spec
from .abstract_channel import AbstractChannel
from .exceptions import (ChannelError, ConsumerCancelled, MessageNacked,
from .protocol import queue_declare_ok_t
def confirm_select(self, nowait=False):
    """Enable publisher confirms for this channel.

        Note: This is an RabbitMQ extension.

        Can now be used if the channel is in transactional mode.

        :param nowait:
            If set, the server will not respond to the method.
            The client should not wait for a reply method. If the
            server could not complete the method it will raise a channel
            or connection exception.
        """
    return self.send_method(spec.Confirm.Select, 'b', (nowait,), wait=None if nowait else spec.Confirm.SelectOk)
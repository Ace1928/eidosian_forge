from __future__ import annotations
import base64
import socket
import sys
import warnings
from array import array
from collections import OrderedDict, defaultdict, namedtuple
from itertools import count
from multiprocessing.util import Finalize
from queue import Empty
from time import monotonic, sleep
from typing import TYPE_CHECKING
from amqp.protocol import queue_declare_ok_t
from kombu.exceptions import ChannelError, ResourceError
from kombu.log import get_logger
from kombu.transport import base
from kombu.utils.div import emergency_dump_state
from kombu.utils.encoding import bytes_to_str, str_to_bytes
from kombu.utils.scheduling import FairCycle
from kombu.utils.uuid import uuid
from .exchange import STANDARD_EXCHANGE_TYPES
def basic_cancel(self, consumer_tag):
    """Cancel consumer by consumer tag."""
    if consumer_tag in self._consumers:
        self._consumers.remove(consumer_tag)
        self._reset_cycle()
        queue = self._tag_to_queue.pop(consumer_tag, None)
        try:
            self._active_queues.remove(queue)
        except ValueError:
            pass
        self.connection._callbacks.pop(queue, None)
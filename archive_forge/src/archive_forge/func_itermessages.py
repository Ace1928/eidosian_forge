from __future__ import annotations
import os
import socket
import threading
from collections import deque
from contextlib import contextmanager
from functools import partial
from itertools import count
from uuid import NAMESPACE_OID, uuid3, uuid4, uuid5
from amqp import ChannelError, RecoverableConnectionError
from .entity import Exchange, Queue
from .log import get_logger
from .serialization import registry as serializers
from .utils.uuid import uuid
def itermessages(conn, channel, queue, limit=1, timeout=None, callbacks=None, **kwargs):
    """Iterator over messages."""
    return drain_consumer(conn.Consumer(queues=[queue], channel=channel, **kwargs), limit=limit, timeout=timeout, callbacks=callbacks)
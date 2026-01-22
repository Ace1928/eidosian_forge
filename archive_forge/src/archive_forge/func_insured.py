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
def insured(pool, fun, args, kwargs, errback=None, on_revive=None, **opts):
    """Function wrapper to handle connection errors.

    Ensures function performing broker commands completes
    despite intermittent connection failures.
    """
    errback = errback or _ensure_errback
    with pool.acquire(block=True) as conn:
        conn.ensure_connection(errback=errback)
        channel = conn.default_channel
        revive = partial(revive_connection, conn, on_revive=on_revive)
        insured = conn.autoretry(fun, channel, errback=errback, on_revive=revive, **opts)
        retval, _ = insured(*args, **dict(kwargs, connection=conn))
        return retval
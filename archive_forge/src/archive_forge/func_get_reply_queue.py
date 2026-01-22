from __future__ import annotations
import socket
import warnings
from collections import defaultdict, deque
from contextlib import contextmanager
from copy import copy
from itertools import count
from time import time
from . import Consumer, Exchange, Producer, Queue
from .clocks import LamportClock
from .common import maybe_declare, oid_from
from .exceptions import InconsistencyError
from .log import get_logger
from .matcher import match
from .utils.functional import maybe_evaluate, reprcall
from .utils.objects import cached_property
from .utils.uuid import uuid
def get_reply_queue(self):
    oid = self.oid
    return Queue(f'{oid}.{self.reply_exchange.name}', exchange=self.reply_exchange, routing_key=oid, durable=False, auto_delete=True, expires=self.reply_queue_expires, message_ttl=self.reply_queue_ttl)
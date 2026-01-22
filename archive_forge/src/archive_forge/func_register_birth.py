import contextlib
import errno
import logging
import math
import os
import random
import signal
import socket
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta
from enum import Enum
from random import shuffle
from types import FrameType
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Type, Union
from uuid import uuid4
from contextlib import suppress
import redis.exceptions
from . import worker_registration
from .command import PUBSUB_CHANNEL_TEMPLATE, handle_command, parse_payload
from .connections import get_current_connection, pop_connection, push_connection
from .defaults import (
from .exceptions import DequeueTimeout, DeserializationError, ShutDownImminentException
from .job import Job, JobStatus
from .logutils import blue, green, setup_loghandlers, yellow
from .maintenance import clean_intermediate_queue
from .queue import Queue
from .registry import StartedJobRegistry, clean_registries
from .scheduler import RQScheduler
from .serializers import resolve_serializer
from .suspension import is_suspended
from .timeouts import HorseMonitorTimeoutException, JobTimeoutException, UnixSignalDeathPenalty
from .utils import as_text, backend_class, compact, ensure_list, get_version, utcformat, utcnow, utcparse
from .version import VERSION
def register_birth(self):
    """Registers its own birth."""
    self.log.debug('Registering birth of worker %s', self.name)
    if self.connection.exists(self.key) and (not self.connection.hexists(self.key, 'death')):
        msg = 'There exists an active worker named {0!r} already'
        raise ValueError(msg.format(self.name))
    key = self.key
    queues = ','.join(self.queue_names())
    with self.connection.pipeline() as p:
        p.delete(key)
        now = utcnow()
        now_in_string = utcformat(now)
        self.birth_date = now
        mapping = {'birth': now_in_string, 'last_heartbeat': now_in_string, 'queues': queues, 'pid': self.pid, 'hostname': self.hostname, 'ip_address': self.ip_address, 'version': self.version, 'python_version': self.python_version}
        if self.get_redis_server_version() >= (4, 0, 0):
            p.hset(key, mapping=mapping)
        else:
            p.hmset(key, mapping)
        worker_registration.register(self, p)
        p.expire(key, self.worker_ttl + 60)
        p.execute()
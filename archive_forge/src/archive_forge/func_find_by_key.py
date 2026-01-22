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
@classmethod
def find_by_key(cls, worker_key: str, connection: Optional['Redis']=None, job_class: Optional[Type['Job']]=None, queue_class: Optional[Type['Queue']]=None, serializer=None) -> 'Worker':
    """Returns a Worker instance, based on the naming conventions for
        naming the internal Redis keys.  Can be used to reverse-lookup Workers
        by their Redis keys.

        Args:
            worker_key (str): The worker key
            connection (Optional[Redis], optional): Redis connection. Defaults to None.
            job_class (Optional[Type[Job]], optional): The job class if custom class is being used. Defaults to None.
            queue_class (Optional[Type[Queue]]): The queue class if a custom class is being used. Defaults to None.
            serializer (Any, optional): The serializer to use. Defaults to None.

        Raises:
            ValueError: If the key doesn't start with `rq:worker:`, the default worker namespace prefix.

        Returns:
            worker (Worker): The Worker instance.
        """
    prefix = cls.redis_worker_namespace_prefix
    if not worker_key.startswith(prefix):
        raise ValueError('Not a valid RQ worker key: %s' % worker_key)
    if connection is None:
        connection = get_current_connection()
    if not connection.exists(worker_key):
        connection.srem(cls.redis_workers_keys, worker_key)
        return None
    name = worker_key[len(prefix):]
    worker = cls([], name, connection=connection, job_class=job_class, queue_class=queue_class, prepare_for_work=False, serializer=serializer)
    worker.refresh()
    return worker
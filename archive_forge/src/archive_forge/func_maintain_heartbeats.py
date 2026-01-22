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
def maintain_heartbeats(self, job: 'Job'):
    """Updates worker and job's last heartbeat field. If job was
        enqueued with `result_ttl=0`, a race condition could happen where this heartbeat
        arrives after job has been deleted, leaving a job key that contains only
        `last_heartbeat` field.

        hset() is used when updating job's timestamp. This command returns 1 if a new
        Redis key is created, 0 otherwise. So in this case we check the return of job's
        heartbeat() command. If a new key was created, this means the job was already
        deleted. In this case, we simply send another delete command to remove the key.

        https://github.com/rq/rq/issues/1450
        """
    with self.connection.pipeline() as pipeline:
        self.heartbeat(self.job_monitoring_interval + 60, pipeline=pipeline)
        ttl = self.get_heartbeat_ttl(job)
        job.heartbeat(utcnow(), ttl, pipeline=pipeline, xx=True)
        results = pipeline.execute()
        if results[2] == 1:
            self.connection.delete(job.key)
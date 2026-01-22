import logging
import os
import signal
import time
import traceback
from datetime import datetime
from enum import Enum
from multiprocessing import Process
from typing import List, Set
from redis import ConnectionPool, Redis
from .connections import parse_connection
from .defaults import DEFAULT_LOGGING_DATE_FORMAT, DEFAULT_LOGGING_FORMAT, DEFAULT_SCHEDULER_FALLBACK_PERIOD
from .job import Job
from .logutils import setup_loghandlers
from .queue import Queue
from .registry import ScheduledJobRegistry
from .serializers import resolve_serializer
from .utils import current_timestamp, parse_names
def prepare_registries(self, queue_names: str=None):
    """Prepare scheduled job registries for use"""
    self._scheduled_job_registries = []
    if not queue_names:
        queue_names = self._acquired_locks
    for name in queue_names:
        self._scheduled_job_registries.append(ScheduledJobRegistry(name, connection=self.connection, serializer=self.serializer))
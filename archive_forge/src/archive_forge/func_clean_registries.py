import calendar
import logging
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, List, Optional, Type, Union
from rq.serializers import resolve_serializer
from .timeouts import BaseDeathPenalty, UnixSignalDeathPenalty
from .connections import resolve_connection
from .defaults import DEFAULT_FAILURE_TTL
from .exceptions import AbandonedJobError, InvalidJobOperation, NoSuchJobError
from .job import Job, JobStatus
from .queue import Queue
from .utils import as_text, backend_class, current_timestamp
def clean_registries(queue: 'Queue'):
    """Cleans StartedJobRegistry, FinishedJobRegistry and FailedJobRegistry of a queue.

    Args:
        queue (Queue): The queue to clean
    """
    registry = FinishedJobRegistry(name=queue.name, connection=queue.connection, job_class=queue.job_class, serializer=queue.serializer)
    registry.cleanup()
    registry = StartedJobRegistry(name=queue.name, connection=queue.connection, job_class=queue.job_class, serializer=queue.serializer)
    registry.cleanup()
    registry = FailedJobRegistry(name=queue.name, connection=queue.connection, job_class=queue.job_class, serializer=queue.serializer)
    registry.cleanup()
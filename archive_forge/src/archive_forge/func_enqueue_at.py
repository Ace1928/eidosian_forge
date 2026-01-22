import logging
import sys
import traceback
import uuid
import warnings
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from functools import total_ordering
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from redis import WatchError
from .timeouts import BaseDeathPenalty, UnixSignalDeathPenalty
from .connections import resolve_connection
from .defaults import DEFAULT_RESULT_TTL
from .dependency import Dependency
from .exceptions import DequeueTimeout, NoSuchJobError
from .job import Callback, Job, JobStatus
from .logutils import blue, green
from .serializers import resolve_serializer
from .types import FunctionReferenceType, JobDependencyType
from .utils import as_text, backend_class, compact, get_version, import_attribute, parse_timeout, utcnow
def enqueue_at(self, datetime: datetime, f, *args, **kwargs):
    """Schedules a job to be enqueued at specified time

        Args:
            datetime (datetime): _description_
            f (_type_): _description_

        Returns:
            _type_: _description_
        """
    f, timeout, description, result_ttl, ttl, failure_ttl, depends_on, job_id, at_front, meta, retry, on_success, on_failure, on_stopped, pipeline, args, kwargs = Queue.parse_args(f, *args, **kwargs)
    job = self.create_job(f, status=JobStatus.SCHEDULED, args=args, kwargs=kwargs, timeout=timeout, result_ttl=result_ttl, ttl=ttl, failure_ttl=failure_ttl, description=description, depends_on=depends_on, job_id=job_id, meta=meta, retry=retry, on_success=on_success, on_failure=on_failure, on_stopped=on_stopped)
    if at_front:
        job.enqueue_at_front = True
    return self.schedule_job(job, datetime, pipeline=pipeline)
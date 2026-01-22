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
def remove_jobs(self, timestamp: Optional[datetime]=None, pipeline: Optional['Pipeline']=None):
    """Remove jobs whose timestamp is in the past from registry.

        Args:
            timestamp (Optional[datetime], optional): The timestamp. Defaults to None.
            pipeline (Optional[Pipeline], optional): The Redis pipeline. Defaults to None.
        """
    connection = pipeline if pipeline is not None else self.connection
    score = timestamp if timestamp is not None else current_timestamp()
    return connection.zremrangebyscore(self.key, 0, score)
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
def get_job_ids(self, start: int=0, end: int=-1):
    """Returns list of all job ids.

        Args:
            start (int, optional): _description_. Defaults to 0.
            end (int, optional): _description_. Defaults to -1.

        Returns:
            _type_: _description_
        """
    self.cleanup()
    return [as_text(job_id) for job_id in self.connection.zrange(self.key, start, end)]
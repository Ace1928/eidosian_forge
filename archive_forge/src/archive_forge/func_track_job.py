from __future__ import annotations
import enum
import typing
import datetime
import croniter
from aiokeydb.v2.types.base import BaseModel, lazyproperty, Field, validator
from aiokeydb.v2.utils.queue import (
from aiokeydb.v2.configs import settings
from aiokeydb.v2.utils.logs import logger
from aiokeydb.v2.types.static import JobStatus, TaskType, TERMINAL_STATUSES, UNSUCCESSFUL_TERMINAL_STATUSES, INCOMPLETE_STATUSES
def track_job(self, job: 'Job'):
    """
        Tracks the job in the function tracker
        """
    if job.status in UNSUCCESSFUL_TERMINAL_STATUSES:
        self.failed_durations.append(job.job_duration)
        self.last_failed = datetime.datetime.now(tz=datetime.timezone.utc)
    elif job.status in TERMINAL_STATUSES:
        self.completed_durations.append(job.job_duration)
        self.last_completed = datetime.datetime.now(tz=datetime.timezone.utc)
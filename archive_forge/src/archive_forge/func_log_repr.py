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
@property
def log_repr(self):
    """
        Shortened representation of the job.
        """
    kwargs = ', '.join((f'{k}={v}' for k, v in {'status': self.status, 'attempts': self.attempts, 'progress': self.progress, 'kwargs': self.short_kwargs, 'scheduled': self.scheduled, 'process_ms': self.duration('process'), 'start_ms': self.duration('start'), 'total_ms': self.duration('total'), 'error': self.error, 'meta': self.meta}.items() if v is not None))
    return f'{kwargs}'
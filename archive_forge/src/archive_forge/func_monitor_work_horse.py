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
def monitor_work_horse(self, job: 'Job', queue: 'Queue'):
    """The worker will monitor the work horse and make sure that it
        either executes successfully or the status of the job is set to
        failed

        Args:
            job (Job): _description_
            queue (Queue): _description_
        """
    retpid = ret_val = rusage = None
    job.started_at = utcnow()
    while True:
        try:
            with self.death_penalty_class(self.job_monitoring_interval, HorseMonitorTimeoutException):
                retpid, ret_val, rusage = self.wait_for_horse()
            break
        except HorseMonitorTimeoutException:
            self.set_current_job_working_time((utcnow() - job.started_at).total_seconds())
            if job.timeout != -1 and self.current_job_working_time > job.timeout + 60:
                self.heartbeat(self.job_monitoring_interval + 60)
                self.kill_horse()
                self.wait_for_horse()
                break
            self.maintain_heartbeats(job)
        except OSError as e:
            if e.errno != errno.EINTR:
                raise
            self.heartbeat()
    self.set_current_job_working_time(0)
    self._horse_pid = 0
    if ret_val == os.EX_OK:
        return
    job_status = job.get_status()
    if job_status is None:
        return
    elif self._stopped_job_id == job.id:
        self.log.warning('Job stopped by user, moving job to FailedJobRegistry')
        if job.stopped_callback:
            job.execute_stopped_callback(self.death_penalty_class)
        self.handle_job_failure(job, queue=queue, exc_string='Job stopped by user, work-horse terminated.')
    elif job_status not in [JobStatus.FINISHED, JobStatus.FAILED]:
        if not job.ended_at:
            job.ended_at = utcnow()
        signal_msg = f' (signal {os.WTERMSIG(ret_val)})' if ret_val and os.WIFSIGNALED(ret_val) else ''
        exc_string = f'Work-horse terminated unexpectedly; waitpid returned {ret_val}{signal_msg}; '
        self.log.warning('Moving job to FailedJobRegistry (%s)', exc_string)
        self.handle_work_horse_killed(job, retpid, ret_val, rusage)
        self.handle_job_failure(job, queue=queue, exc_string=exc_string)
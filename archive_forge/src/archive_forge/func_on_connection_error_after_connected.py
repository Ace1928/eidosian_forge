from the broker, processing the messages and keeping the broker connections
import errno
import logging
import os
import warnings
from collections import defaultdict
from time import sleep
from billiard.common import restart_state
from billiard.exceptions import RestartFreqExceeded
from kombu.asynchronous.semaphore import DummyLock
from kombu.exceptions import ContentDisallowed, DecodeError
from kombu.utils.compat import _detect_environment
from kombu.utils.encoding import safe_repr
from kombu.utils.limits import TokenBucket
from vine import ppartial, promise
from celery import bootsteps, signals
from celery.app.trace import build_tracer
from celery.exceptions import (CPendingDeprecationWarning, InvalidTaskError, NotRegistered, WorkerShutdown,
from celery.utils.functional import noop
from celery.utils.log import get_logger
from celery.utils.nodenames import gethostname
from celery.utils.objects import Bunch
from celery.utils.text import truncate
from celery.utils.time import humanize_seconds, rate
from celery.worker import loops
from celery.worker.state import active_requests, maybe_shutdown, requests, reserved_requests, task_reserved
def on_connection_error_after_connected(self, exc):
    warn(CONNECTION_RETRY, exc_info=True)
    try:
        self.connection.collect()
    except Exception:
        pass
    if self.app.conf.worker_cancel_long_running_tasks_on_connection_loss:
        for request in tuple(active_requests):
            if request.task.acks_late and (not request.acknowledged):
                warn(TERMINATING_TASK_ON_RESTART_AFTER_A_CONNECTION_LOSS, request)
                request.cancel(self.pool)
    else:
        warnings.warn(CANCEL_TASKS_BY_DEFAULT, CPendingDeprecationWarning)
    if self.app.conf.worker_enable_prefetch_count_reduction:
        self.initial_prefetch_count = max(self.prefetch_multiplier, self.max_prefetch_count - len(tuple(active_requests)) * self.prefetch_multiplier)
        self._maximum_prefetch_restored = self.initial_prefetch_count == self.max_prefetch_count
        if not self._maximum_prefetch_restored:
            logger.info(f'Temporarily reducing the prefetch count to {self.initial_prefetch_count} to avoid over-fetching since {len(tuple(active_requests))} tasks are currently being processed.\nThe prefetch count will be gradually restored to {self.max_prefetch_count} as the tasks complete processing.')
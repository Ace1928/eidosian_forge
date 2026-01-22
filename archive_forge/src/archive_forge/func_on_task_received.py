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
def on_task_received(message):
    payload = None
    try:
        type_ = message.headers['task']
    except TypeError:
        return on_unknown_message(None, message)
    except KeyError:
        try:
            payload = message.decode()
        except Exception as exc:
            return self.on_decode_error(message, exc)
        try:
            type_, payload = (payload['task'], payload)
        except (TypeError, KeyError):
            return on_unknown_message(payload, message)
    try:
        strategy = strategies[type_]
    except KeyError as exc:
        return on_unknown_task(None, message, exc)
    else:
        try:
            ack_log_error_promise = promise(call_soon, (message.ack_log_error,), on_error=self._restore_prefetch_count_after_connection_restart)
            reject_log_error_promise = promise(call_soon, (message.reject_log_error,), on_error=self._restore_prefetch_count_after_connection_restart)
            if not self._maximum_prefetch_restored and self.restart_count > 0 and (self._new_prefetch_count <= self.max_prefetch_count):
                ack_log_error_promise.then(self._restore_prefetch_count_after_connection_restart, on_error=self._restore_prefetch_count_after_connection_restart)
                reject_log_error_promise.then(self._restore_prefetch_count_after_connection_restart, on_error=self._restore_prefetch_count_after_connection_restart)
            strategy(message, payload, ack_log_error_promise, reject_log_error_promise, callbacks)
        except (InvalidTaskError, ContentDisallowed) as exc:
            return on_invalid_task(payload, message, exc)
        except DecodeError as exc:
            return self.on_decode_error(message, exc)
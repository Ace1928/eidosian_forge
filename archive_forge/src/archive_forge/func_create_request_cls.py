import logging
import sys
from datetime import datetime
from time import monotonic, time
from weakref import ref
from billiard.common import TERM_SIGNAME
from billiard.einfo import ExceptionWithTraceback
from kombu.utils.encoding import safe_repr, safe_str
from kombu.utils.objects import cached_property
from celery import current_app, signals
from celery.app.task import Context
from celery.app.trace import fast_trace_task, trace_task, trace_task_ret
from celery.concurrency.base import BasePool
from celery.exceptions import (Ignore, InvalidTaskError, Reject, Retry, TaskRevokedError, Terminated,
from celery.platforms import signals as _signals
from celery.utils.functional import maybe, maybe_list, noop
from celery.utils.log import get_logger
from celery.utils.nodenames import gethostname
from celery.utils.serialization import get_pickled_exception
from celery.utils.time import maybe_iso8601, maybe_make_aware, timezone
from . import state
def create_request_cls(base, task, pool, hostname, eventer, ref=ref, revoked_tasks=revoked_tasks, task_ready=task_ready, trace=None, app=current_app):
    default_time_limit = task.time_limit
    default_soft_time_limit = task.soft_time_limit
    apply_async = pool.apply_async
    acks_late = task.acks_late
    events = eventer and eventer.enabled
    if trace is None:
        trace = fast_trace_task if app.use_fast_trace_task else trace_task_ret

    class Request(base):

        def execute_using_pool(self, pool, **kwargs):
            task_id = self.task_id
            if self.revoked():
                raise TaskRevokedError(task_id)
            time_limit, soft_time_limit = self.time_limits
            result = apply_async(trace, args=(self.type, task_id, self.request_dict, self.body, self.content_type, self.content_encoding), accept_callback=self.on_accepted, timeout_callback=self.on_timeout, callback=self.on_success, error_callback=self.on_failure, soft_timeout=soft_time_limit or default_soft_time_limit, timeout=time_limit or default_time_limit, correlation_id=task_id)
            self._apply_result = maybe(ref, result)
            return result

        def on_success(self, failed__retval__runtime, **kwargs):
            failed, retval, runtime = failed__retval__runtime
            if failed:
                exc = retval.exception
                if isinstance(exc, ExceptionWithTraceback):
                    exc = exc.exc
                if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                    raise exc
                return self.on_failure(retval, return_ok=True)
            task_ready(self)
            if acks_late:
                self.acknowledge()
            if events:
                self.send_event('task-succeeded', result=retval, runtime=runtime)
    return Request
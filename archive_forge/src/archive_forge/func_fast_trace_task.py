import logging
import os
import sys
import time
from collections import namedtuple
from warnings import warn
from billiard.einfo import ExceptionInfo, ExceptionWithTraceback
from kombu.exceptions import EncodeError
from kombu.serialization import loads as loads_message
from kombu.serialization import prepare_accept_content
from kombu.utils.encoding import safe_repr, safe_str
from celery import current_app, group, signals, states
from celery._state import _task_stack
from celery.app.task import Context
from celery.app.task import Task as BaseTask
from celery.exceptions import BackendGetMetaError, Ignore, InvalidTaskError, Reject, Retry
from celery.result import AsyncResult
from celery.utils.log import get_logger
from celery.utils.nodenames import gethostname
from celery.utils.objects import mro_lookup
from celery.utils.saferepr import saferepr
from celery.utils.serialization import get_pickleable_etype, get_pickleable_exception, get_pickled_exception
from celery.worker.state import successful_requests
def fast_trace_task(task, uuid, request, body, content_type, content_encoding, loads=loads_message, _loc=None, hostname=None, **_):
    _loc = _localized if not _loc else _loc
    embed = None
    tasks, accept, hostname = _loc
    if content_type:
        args, kwargs, embed = loads(body, content_type, content_encoding, accept=accept)
    else:
        args, kwargs, embed = body
    request.update({'args': args, 'kwargs': kwargs, 'hostname': hostname, 'is_eager': False}, **embed or {})
    R, I, T, Rstr = tasks[task].__trace__(uuid, args, kwargs, request)
    return (1, R, T) if I else (0, Rstr, T)
import sys
import time
import warnings
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from functools import partial
from weakref import WeakValueDictionary
from billiard.einfo import ExceptionInfo
from kombu.serialization import dumps, loads, prepare_accept_content
from kombu.serialization import registry as serializer_registry
from kombu.utils.encoding import bytes_to_str, ensure_bytes
from kombu.utils.url import maybe_sanitize_url
import celery.exceptions
from celery import current_app, group, maybe_signature, states
from celery._state import get_current_task
from celery.app.task import Context
from celery.exceptions import (BackendGetMetaError, BackendStoreError, ChordError, ImproperlyConfigured,
from celery.result import GroupResult, ResultBase, ResultSet, allow_join_result, result_from_tuple
from celery.utils.collections import BufferMap
from celery.utils.functional import LRUCache, arity_greater
from celery.utils.log import get_logger
from celery.utils.serialization import (create_exception_cls, ensure_serializable, get_pickleable_exception,
from celery.utils.time import get_exponential_backoff_interval
def iter_native(self, result, timeout=None, interval=0.5, no_ack=True, on_message=None, on_interval=None):
    self._ensure_not_eager()
    results = result.results
    if not results:
        return
    task_ids = set()
    for result in results:
        if isinstance(result, ResultSet):
            yield (result.id, result.results)
        else:
            task_ids.add(result.id)
    yield from self.get_many(task_ids, timeout=timeout, interval=interval, no_ack=no_ack, on_message=on_message, on_interval=on_interval)
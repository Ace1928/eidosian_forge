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
def get_task_meta(self, task_id, cache=True):
    """Get task meta from backend.

        if always_retry_backend_operation is activated, in the event of a recoverable exception,
        then retry operation with an exponential backoff until a limit has been reached.
        """
    self._ensure_not_eager()
    if cache:
        try:
            return self._cache[task_id]
        except KeyError:
            pass
    retries = 0
    while True:
        try:
            meta = self._get_task_meta_for(task_id)
            break
        except Exception as exc:
            if self.always_retry and self.exception_safe_to_retry(exc):
                if retries < self.max_retries:
                    retries += 1
                    sleep_amount = get_exponential_backoff_interval(self.base_sleep_between_retries_ms, retries, self.max_sleep_between_retries_ms, True) / 1000
                    self._sleep(sleep_amount)
                else:
                    raise_with_context(BackendGetMetaError('failed to get meta', task_id=task_id))
            else:
                raise
    if cache and meta.get('status') == states.SUCCESS:
        self._cache[task_id] = meta
    return meta
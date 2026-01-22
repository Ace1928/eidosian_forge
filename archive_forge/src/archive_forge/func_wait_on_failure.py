import collections
import contextlib
import os
import re
import threading
import time
import weakref
from six.moves import queue
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.distribute.coordinator import metric_utils
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.distribute.coordinator import utils
from tensorflow.python.distribute.coordinator import values as values_lib
from tensorflow.python.distribute.coordinator import watchdog
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@contextlib.contextmanager
def wait_on_failure(self, on_failure_fn=None, on_transient_failure_fn=None, on_recovery_fn=None, worker_device_name='(unknown)'):
    """Catches worker preemption error and wait until failed workers are back.

    Args:
      on_failure_fn: an optional function to run if preemption happens.
      on_transient_failure_fn: an optional function to run if transient failure
        happens.
      on_recovery_fn: an optional function to run when a worker is recovered
        from preemption.
      worker_device_name: the device name of the worker instance that is passing
        through the failure.

    Yields:
      None.
    """
    assert self._should_preemption_thread_run
    try:
        yield
    except (errors.OpError, ClosureInputError, ClosureAbortedError, TypeError) as e:
        if self._cluster._record_and_ignore_transient_ps_failure(e):
            logging.error('Remote function on worker %s failed with %r:%s\nIt is treated as a transient connectivity failure for now.', worker_device_name, e, e)
            if on_transient_failure_fn:
                on_transient_failure_fn()
            return
        if self._cluster._record_and_ignore_transient_timeouts(e):
            logging.error('Remote function on worker %s failed with %r:%s\nThis derived error is ignored and not reported to users.', worker_device_name, e, e)
            if on_transient_failure_fn:
                on_transient_failure_fn()
            return
        if isinstance(e, errors.CancelledError) and '/job:' in str(e):
            logging.error('Remote function on worker %s failed with %r:%s\nThis derived error is ignored and not reported to users.', worker_device_name, e, e)
            if on_transient_failure_fn:
                on_transient_failure_fn()
            return
        self._validate_preemption_failure(e)
        logging.error('Worker %s failed with %r:%s', worker_device_name, e, e)
        if on_failure_fn:
            on_failure_fn(e)
        with self._cluster_update_lock:
            self._cluster_due_for_update_or_finish.set()
            self._worker_up_cond.wait(_WORKER_MAXIMUM_RECOVERY_SEC)
            if self._error_from_recovery:
                try:
                    raise self._error_from_recovery
                finally:
                    self._error_from_recovery = None
            logging.info('Worker %s has been recovered.', worker_device_name)
        if on_recovery_fn:
            logging.info('Worker %s calling on_recovery_fn', worker_device_name)
            with self.wait_on_failure(on_recovery_fn=on_recovery_fn, on_transient_failure_fn=on_transient_failure_fn, worker_device_name=worker_device_name):
                on_recovery_fn()
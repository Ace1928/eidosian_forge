import time
import numpy as np
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
def wait_for_session(self, master, config=None, max_wait_secs=float('Inf')):
    """Creates a new `Session` and waits for model to be ready.

    Creates a new `Session` on 'master'.  Waits for the model to be
    initialized or recovered from a checkpoint.  It's expected that
    another thread or process will make the model ready, and that this
    is intended to be used by threads/processes that participate in a
    distributed training configuration where a different thread/process
    is responsible for initializing or recovering the model being trained.

    NB: The amount of time this method waits for the session is bounded
    by max_wait_secs. By default, this function will wait indefinitely.

    Args:
      master: `String` representation of the TensorFlow master to use.
      config: Optional ConfigProto proto used to configure the session.
      max_wait_secs: Maximum time to wait for the session to become available.

    Returns:
      A `Session`. May be None if the operation exceeds the timeout
      specified by config.operation_timeout_in_ms.

    Raises:
      tf.DeadlineExceededError: if the session is not available after
        max_wait_secs.
    """
    self._target = master
    if max_wait_secs is None:
        max_wait_secs = float('Inf')
    timer = _CountDownTimer(max_wait_secs)
    while True:
        sess = session.Session(self._target, graph=self._graph, config=config)
        not_ready_msg = None
        not_ready_local_msg = None
        local_init_success, not_ready_local_msg = self._try_run_local_init_op(sess)
        if local_init_success:
            is_ready, not_ready_msg = self._model_ready(sess)
            if is_ready:
                return sess
        self._safe_close(sess)
        remaining_ms_after_wait = timer.secs_remaining() - self._recovery_wait_secs
        if remaining_ms_after_wait < 0:
            raise errors.DeadlineExceededError(None, None, 'Session was not ready after waiting %d secs.' % (max_wait_secs,))
        logging.info('Waiting for model to be ready.  Ready_for_local_init_op:  %s, ready: %s', not_ready_local_msg, not_ready_msg)
        time.sleep(self._recovery_wait_secs)
import time
import numpy as np
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
def prepare_session(self, master, init_op=None, saver=None, checkpoint_dir=None, checkpoint_filename_with_path=None, wait_for_checkpoint=False, max_wait_secs=7200, config=None, init_feed_dict=None, init_fn=None):
    """Creates a `Session`. Makes sure the model is ready to be used.

    Creates a `Session` on 'master'. If a `saver` object is passed in, and
    `checkpoint_dir` points to a directory containing valid checkpoint
    files, then it will try to recover the model from checkpoint. If
    no checkpoint files are available, and `wait_for_checkpoint` is
    `True`, then the process would check every `recovery_wait_secs`,
    up to `max_wait_secs`, for recovery to succeed.

    If the model cannot be recovered successfully then it is initialized by
    running the `init_op` and calling `init_fn` if they are provided.
    The `local_init_op` is also run after init_op and init_fn, regardless of
    whether the model was recovered successfully, but only if
    `ready_for_local_init_op` passes.

    If the model is recovered from a checkpoint it is assumed that all
    global variables have been initialized, in particular neither `init_op`
    nor `init_fn` will be executed.

    It is an error if the model cannot be recovered and no `init_op`
    or `init_fn` or `local_init_op` are passed.

    Args:
      master: `String` representation of the TensorFlow master to use.
      init_op: Optional `Operation` used to initialize the model.
      saver: A `Saver` object used to restore a model.
      checkpoint_dir: Path to the checkpoint files. The latest checkpoint in the
        dir will be used to restore.
      checkpoint_filename_with_path: Full file name path to the checkpoint file.
      wait_for_checkpoint: Whether to wait for checkpoint to become available.
      max_wait_secs: Maximum time to wait for checkpoints to become available.
      config: Optional `ConfigProto` proto used to configure the session.
      init_feed_dict: Optional dictionary that maps `Tensor` objects to feed
        values.  This feed dictionary is passed to the session `run()` call when
        running the init op.
      init_fn: Optional callable used to initialize the model. Called after the
        optional `init_op` is called.  The callable must accept one argument,
        the session being initialized.

    Returns:
      A `Session` object that can be used to drive the model.

    Raises:
      RuntimeError: If the model cannot be initialized or recovered.
      ValueError: If both checkpoint_dir and checkpoint_filename_with_path are
        set.
    """
    sess, is_loaded_from_checkpoint = self._restore_checkpoint(master, saver, checkpoint_dir=checkpoint_dir, checkpoint_filename_with_path=checkpoint_filename_with_path, wait_for_checkpoint=wait_for_checkpoint, max_wait_secs=max_wait_secs, config=config)
    if not is_loaded_from_checkpoint:
        if init_op is None and (not init_fn) and (self._local_init_op is None):
            raise RuntimeError('Model is not initialized and no init_op or init_fn or local_init_op was given')
        if init_op is not None:
            sess.run(init_op, feed_dict=init_feed_dict)
        if init_fn:
            init_fn(sess)
    local_init_success, msg = self._try_run_local_init_op(sess)
    if not local_init_success:
        raise RuntimeError('Init operations did not make model ready for local_init.  Init op: %s, init fn: %s, error: %s' % (_maybe_name(init_op), init_fn, msg))
    is_ready, msg = self._model_ready(sess)
    if not is_ready:
        raise RuntimeError('Init operations did not make model ready.  Init op: %s, init fn: %s, local_init_op: %s, error: %s' % (_maybe_name(init_op), init_fn, self._local_init_op, msg))
    return sess
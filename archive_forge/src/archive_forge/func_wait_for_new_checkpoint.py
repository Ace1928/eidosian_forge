from collections import abc
import os
import time
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util.tf_export import tf_export
def wait_for_new_checkpoint(checkpoint_dir, last_checkpoint=None, seconds_to_sleep=1, timeout=None):
    """Waits until a new checkpoint file is found.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    last_checkpoint: The last checkpoint path used or `None` if we're expecting
      a checkpoint for the first time.
    seconds_to_sleep: The number of seconds to sleep for before looking for a
      new checkpoint.
    timeout: The maximum number of seconds to wait. If left as `None`, then the
      process will wait indefinitely.

  Returns:
    a new checkpoint path, or None if the timeout was reached.
  """
    logging.info('Waiting for new checkpoint at %s', checkpoint_dir)
    stop_time = time.time() + timeout if timeout is not None else None
    while True:
        checkpoint_path = checkpoint_management.latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None or checkpoint_path == last_checkpoint:
            if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
                return None
            time.sleep(seconds_to_sleep)
        else:
            logging.info('Found new checkpoint at %s', checkpoint_path)
            return checkpoint_path
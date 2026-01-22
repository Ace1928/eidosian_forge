import time
import numpy as np
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
Tries to run _local_init_op, if not None, and is ready for local init.

    Args:
      sess: A `Session`.

    Returns:
      A tuple (is_successful, msg), where is_successful is True if
      _local_init_op is None, or we ran _local_init_op, and False otherwise;
      and msg is a `String` with the reason why the model was not ready to run
      local init.
    
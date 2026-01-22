import collections
import os.path
import re
import time
from google.protobuf import text_format
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def checkpoint_exists_internal(checkpoint_prefix):
    """Checks whether a V1 or V2 checkpoint exists with the specified prefix.

  This is an internal function to check if a checkpoint exists,
  since it takes into account the naming difference between V1 and V2 formats.

  Args:
    checkpoint_prefix: the prefix of a V1 or V2 checkpoint, with V2 taking
      priority.  Typically the result of `Saver.save()` or that of
      `tf.train.latest_checkpoint()`, regardless of sharded/non-sharded or
      V1/V2.
  Returns:
    A bool, true if a checkpoint referred to by `checkpoint_prefix` exists.
  """
    pathname = _prefix_to_checkpoint_path(checkpoint_prefix, saver_pb2.SaverDef.V2)
    if file_io.get_matching_files(pathname):
        return True
    elif file_io.get_matching_files(checkpoint_prefix):
        return True
    else:
        return False
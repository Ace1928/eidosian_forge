from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.summary import summary
from tensorflow.python.training import queue_runner
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export('io.match_filenames_once', v1=['io.match_filenames_once', 'train.match_filenames_once'])
@deprecation.deprecated_endpoints('train.match_filenames_once')
def match_filenames_once(pattern, name=None):
    """Save the list of files matching pattern, so it is only computed once.

  NOTE: The order of the files returned is deterministic.

  Args:
    pattern: A file pattern (glob), or 1D tensor of file patterns.
    name: A name for the operations (optional).

  Returns:
    A variable that is initialized to the list of files matching the pattern(s).
  """
    with ops.name_scope(name, 'matching_filenames', [pattern]) as name:
        return variable_v1.VariableV1(name=name, initial_value=io_ops.matching_files(pattern), trainable=False, validate_shape=False, collections=[ops.GraphKeys.LOCAL_VARIABLES])
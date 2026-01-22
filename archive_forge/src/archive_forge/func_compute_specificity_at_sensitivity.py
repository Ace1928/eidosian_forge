from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def compute_specificity_at_sensitivity(tp, tn, fp, fn, name):
    """Computes the specificity at the given sensitivity.

      Args:
        tp: True positives.
        tn: True negatives.
        fp: False positives.
        fn: False negatives.
        name: The name of the operation.

      Returns:
        The specificity using the aggregated values.
      """
    sensitivities = math_ops.divide(tp, tp + fn + kepsilon)
    min_val = math_ops.reduce_min(math_ops.abs(sensitivities - sensitivity))
    indices_at_minval = math_ops.equal(math_ops.abs(sensitivities - sensitivity), min_val)
    indices_at_minval = math_ops.cast(indices_at_minval, dtypes.int64)
    indices_at_minval = math_ops.cumsum(indices_at_minval)
    tf_index = math_ops.argmax(indices_at_minval, 0)
    tf_index = math_ops.cast(tf_index, dtypes.int32)
    return math_ops.divide(tn[tf_index], tn[tf_index] + fp[tf_index] + kepsilon, name)
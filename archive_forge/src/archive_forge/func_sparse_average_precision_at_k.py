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
@tf_export(v1=['metrics.sparse_average_precision_at_k'])
@deprecated(None, 'Use average_precision_at_k instead')
def sparse_average_precision_at_k(labels, predictions, k, weights=None, metrics_collections=None, updates_collections=None, name=None):
    """Renamed to `average_precision_at_k`, please use that method instead."""
    return average_precision_at_k(labels=labels, predictions=predictions, k=k, weights=weights, metrics_collections=metrics_collections, updates_collections=updates_collections, name=name)
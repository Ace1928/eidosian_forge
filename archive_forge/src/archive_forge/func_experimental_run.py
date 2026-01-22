import collections
import contextlib
import copy
import enum  # pylint: disable=g-bad-import-order
import functools
import threading
import weakref
import six
from tensorflow.python import tf2
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context as eager_context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import tape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import distribute as ds_types
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@deprecated(None, 'This method is not available in TF 2.x. Please switch to using `run` instead.')
def experimental_run(self, fn, input_iterator=None):
    """Runs ops in `fn` on each replica, with inputs from `input_iterator`.

    DEPRECATED: This method is not available in TF 2.x. Please switch
    to using `run` instead.

    When eager execution is enabled, executes ops specified by `fn` on each
    replica. Otherwise, builds a graph to execute the ops on each replica.

    Each replica will take a single, different input from the inputs provided by
    one `get_next` call on the input iterator.

    `fn` may call `tf.distribute.get_replica_context()` to access members such
    as `replica_id_in_sync_group`.

    IMPORTANT: Depending on the `tf.distribute.Strategy` implementation being
    used, and whether eager execution is enabled, `fn` may be called one or more
    times (once for each replica).

    Args:
      fn: The function to run. The inputs to the function must match the outputs
        of `input_iterator.get_next()`. The output must be a `tf.nest` of
        `Tensor`s.
      input_iterator: (Optional) input iterator from which the inputs are taken.

    Returns:
      Merged return value of `fn` across replicas. The structure of the return
      value is the same as the return value from `fn`. Each element in the
      structure can either be `PerReplica` (if the values are unsynchronized),
      `Mirrored` (if the values are kept in sync), or `Tensor` (if running on a
      single replica).
    """
    return super(StrategyV1, self).experimental_run(fn, input_iterator)
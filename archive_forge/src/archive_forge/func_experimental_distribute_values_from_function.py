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
def experimental_distribute_values_from_function(self, value_fn):
    """Generates `tf.distribute.DistributedValues` from `value_fn`.

    This function is to generate `tf.distribute.DistributedValues` to pass
    into `run`, `reduce`, or other methods that take
    distributed values when not using datasets.

    Args:
      value_fn: The function to run to generate values. It is called for
        each replica with `tf.distribute.ValueContext` as the sole argument. It
        must return a Tensor or a type that can be converted to a Tensor.
    Returns:
      A `tf.distribute.DistributedValues` containing a value for each replica.

    Example usage:

    1.  Return constant value per replica:

        >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
        >>> def value_fn(ctx):
        ...   return tf.constant(1.)
        >>> distributed_values = (
        ...     strategy.experimental_distribute_values_from_function(
        ...        value_fn))
        >>> local_result = strategy.experimental_local_results(
        ...     distributed_values)
        >>> local_result
        (<tf.Tensor: shape=(), dtype=float32, numpy=1.0>,
        <tf.Tensor: shape=(), dtype=float32, numpy=1.0>)

    2.  Distribute values in array based on replica_id: {: value=2}

        >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
        >>> array_value = np.array([3., 2., 1.])
        >>> def value_fn(ctx):
        ...   return array_value[ctx.replica_id_in_sync_group]
        >>> distributed_values = (
        ...     strategy.experimental_distribute_values_from_function(
        ...         value_fn))
        >>> local_result = strategy.experimental_local_results(
        ...     distributed_values)
        >>> local_result
        (3.0, 2.0)

    3.  Specify values using num_replicas_in_sync:  {: value=3}

        >>> strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
        >>> def value_fn(ctx):
        ...   return ctx.num_replicas_in_sync
        >>> distributed_values = (
        ...     strategy.experimental_distribute_values_from_function(
        ...         value_fn))
        >>> local_result = strategy.experimental_local_results(
        ...     distributed_values)
        >>> local_result
        (2, 2)

    4.  Place values on devices and distribute: {: value=4}

        ```
        strategy = tf.distribute.TPUStrategy()
        worker_devices = strategy.extended.worker_devices
        multiple_values = []
        for i in range(strategy.num_replicas_in_sync):
          with tf.device(worker_devices[i]):
            multiple_values.append(tf.constant(1.0))

        def value_fn(ctx):
          return multiple_values[ctx.replica_id_in_sync_group]

        distributed_values = strategy.
          experimental_distribute_values_from_function(
          value_fn)
        ```

    """
    return self._extended._experimental_distribute_values_from_function(value_fn)
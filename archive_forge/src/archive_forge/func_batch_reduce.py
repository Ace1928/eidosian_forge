import collections
import copy
import multiprocessing.dummy
import multiprocessing.pool
import threading
import numpy as np
import six
from tensorflow.python.client import device_lib
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import kernels
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def batch_reduce(self, reduce_op, value_destination_pairs, options=None):
    """Reduce values to destinations in batches.

    See `tf.distribute.StrategyExtended.batch_reduce_to`. This can only be
    called in the cross-replica context.

    Args:
      reduce_op: a `tf.distribute.ReduceOp` specifying how values should be
        combined.
      value_destination_pairs: a sequence of (value, destinations) pairs. See
        `tf.distribute.CrossDeviceOps.reduce` for descriptions.
      options: a `tf.distribute.experimental.CommunicationOptions`. See
        `tf.distribute.experimental.CommunicationOptions` for details.

    Returns:
      A list of `tf.Tensor` or `tf.distribute.DistributedValues`, one per pair
      in `value_destination_pairs`.

    Raises:
      ValueError: if `value_destination_pairs` is not an iterable of
        tuples of `tf.distribute.DistributedValues` and destinations.
    """
    if options is None:
        options = collective_util.Options()
    if not _validate_value_destination_pairs(value_destination_pairs):
        value_destination_pairs = _normalize_value_destination_pairs(value_destination_pairs)
    for _, d in value_destination_pairs:
        validate_destinations(d)
    if self._num_between_graph_workers == 1 and _all_devices_match(value_destination_pairs, self._canonicalize_devices) and (len(value_destination_pairs[0][0].values) == 1):
        return [distribute_utils.regroup(v.values, wrap_class=value_lib.Mirrored) for v, _ in value_destination_pairs]
    if options is None:
        options = collective_util.Options()
    return self.batch_reduce_implementation(reduce_op, value_destination_pairs, options)
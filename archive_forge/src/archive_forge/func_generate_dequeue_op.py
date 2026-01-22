import itertools
import numpy as np
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.tpu import tpu_name_util
from tensorflow.python.tpu import tpu_sharding
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import nest
def generate_dequeue_op(self, tpu_device=0):
    """Generate TPU dequeue ops.

    Args:
      tpu_device: The TPU device ordinal where the infeed instruction should be
        placed.

    Returns:
      A list of Outputs corresponding to a partition of infeed dequeued
      into XLA, suitable for use within a replicated block.

    Raises:
      ValueError: if the types or shapes of the tuple elements have not been
      set; or if a dequeue op has already been generated.
    """
    self.freeze()
    if self._generated_dequeue_op and (not ops.inside_function()):
        raise ValueError("Can't generate two dequeue Ops from the same queue")
    self._generated_dequeue_op = True
    full_name = '%s/dequeue' % self._name
    sharded_shapes = [policy.get_sharded_shape(shape) for shape, policy in zip(self._tuple_shapes, self._sharding_policies)]
    with ops.device(tpu_name_util.core(tpu_device)):
        values = tpu_ops.infeed_dequeue_tuple(dtypes=self._tuple_types, shapes=sharded_shapes, name=full_name)
    return tag_sharding_attribute_for_dequeued_tensors(values, self._input_partition_dims)
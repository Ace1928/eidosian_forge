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
def set_shard_dimensions(self, shard_dimensions):
    """Sets the shard_dimension of each element of the queue.

    shard_dimensions must be a list of length
    self.number_of_tuple_elements, and each element must be
    convertible to a Dimension compatible with self.tuple_shapes.

    Args:
      shard_dimensions: the dimensions of each queue element.

    Raises:
      ValueError: if shard_dimensions is not of length
        self.number_of_tuple_elements; or an element of
        shard_dimensions cannot be converted to a Dimension; or an
        element of shard_dimensions is a Dimension that is out of
        range for the corresponding tuple element shape.
    """
    if len(shard_dimensions) != self.number_of_tuple_elements:
        raise ValueError(f'shard_dimensions is {str(shard_dimensions)}, but must be a list of length {self.number_of_tuple_elements}')
    for policy, dimension in zip(self._sharding_policies, shard_dimensions):
        policy.set_shard_dimension(dimension)
    self._validate()
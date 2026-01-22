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
@property
def shard_dimensions(self):
    """Gets the shard dimension of each tuple element.

    Returns:
      A list of length number_of_tuple_elements, where each list entry
      is the shard dimension of that tuple element or None if the
      shard dimension has not been set.
    """
    return [policy.shard_dimension for policy in self._sharding_policies]
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
def set_number_of_shards(self, number_of_shards):
    """Sets the number of shards to use for the InfeedQueue.

    Args:
      number_of_shards: number of ways to shard the InfeedQueue.

    Raises:
      ValueError: if number_of_shards is not > 0; or the policies have
        been frozen and number_of_shards was already set to something
        else.
    """
    for policy in self._sharding_policies:
        policy.set_number_of_shards(number_of_shards)
        policy.set_number_of_partitions(self._number_of_partitions)
    self._validate()
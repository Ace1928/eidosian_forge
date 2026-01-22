from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def make_seeds(self, count=1):
    """Generates seeds for stateless random ops.

    For example:

    ```python
    seeds = get_global_generator().make_seeds(count=10)
    for i in range(10):
      seed = seeds[:, i]
      numbers = stateless_random_normal(shape=[2, 3], seed=seed)
      ...
    ```

    Args:
      count: the number of seed pairs (note that stateless random ops need a
        pair of seeds to invoke).

    Returns:
      A tensor of shape [2, count] and dtype int64.
    """
    alg = self.algorithm
    if alg in (a.value for a in random_ops_util.Algorithm):
        keys = self._make_int64_keys(shape=[count])
        zeros = array_ops.zeros_like(keys)
        return array_ops_stack.stack([keys, zeros])
    else:
        raise ValueError(stateless_random_ops.unsupported_alg_error_msg(alg))
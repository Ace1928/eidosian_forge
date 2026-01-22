import functools
from typing import Any, Callable, Dict, Iterable, List, Optional, Text, Tuple, Union
from absl import logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import save_context
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def make_sharded_variable_creator(hosts: List[Text]) -> Callable[..., TPUEmbeddingVariable]:
    """Makes a sharded variable creator given a list of hosts.

  Args:
    hosts: a list of tensorflow devices on which to shard the tensors.

  Returns:
    A variable creator function.
  """

    def sharded_variable_creator(next_creator: Callable[..., tf_variables.Variable], *args, **kwargs):
        """The sharded variable creator."""
        kwargs['skip_mirrored_creator'] = True
        num_hosts = len(hosts)
        name, shape, dtype, unwrapped_initial_value = extract_variable_info(kwargs)
        initial_value = kwargs['initial_value']
        rows = shape[0]
        cols = shape[1]
        partial_partition = rows % num_hosts
        full_rows_per_host = rows // num_hosts
        partitions = [full_rows_per_host + 1] * partial_partition + [full_rows_per_host] * (num_hosts - partial_partition)
        variables = []
        sharding_aware = 'shard_info' in tf_inspect.getargspec(initial_value).args
        offset = 0
        kwargs['dtype'] = dtype
        for i, p in enumerate(partitions):
            if p == 0:
                continue
            with ops.device(hosts[i]):
                kwargs['name'] = '{}_{}'.format(name, i)
                kwargs['shape'] = (p, cols)
                if sharding_aware:
                    shard_info = base.ShardInfo(kwargs['shape'], (offset, 0))
                    kwargs['initial_value'] = functools.partial(initial_value, shard_info=shard_info)
                    offset += p
                else:
                    kwargs['initial_value'] = functools.partial(unwrapped_initial_value, kwargs['shape'], dtype=dtype)
                variables.append(next_creator(*args, **kwargs))
        return TPUEmbeddingVariable(variables, name=name)
    return sharded_variable_creator
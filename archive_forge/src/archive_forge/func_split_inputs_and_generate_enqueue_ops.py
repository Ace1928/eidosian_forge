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
def split_inputs_and_generate_enqueue_ops(self, inputs, device_assignment=None, placement_function=None, tpu_ordinal_function=None):
    """POORLY-PERFORMING ON MULTI-HOST SYSTEMS.

    Generates the host-side Ops to enqueue a tuple.

    This method performs poorly because it takes an entire input on a single
    host, splits it, and distributes it to all of the cores. It is present only
    to simplify tutorial examples.

    inputs is a list of Tensors to use to feed the queue. Each input is split
    into self.number_of_shards shards. Returns an Op for each shard to enqueue
    the shard. The Op for shard i is placed on device placement_function(i).

    Implicitly freezes the queue configuration if it is not already
    frozen. If the configuration has already been frozen, and is not
    compatible with the types and shapes of inputs, an error
    will be raised.

    Args:
      inputs: a list of Tensors which indicates the types and shapes of the
        queue tuple.
     device_assignment: if not `None`, a TPU `DeviceAssignment`. If
        device_assignment is not `None`, but `placement_function` and
        `ordinal_function` are None, then `device_assignment` will be used to
        place infeeds on the first k TPU shards, where k is the number of shards
        in the queue. If all three are `None`, then default placement and
        ordinal functions are used.
      placement_function: if not None, a function that takes the shard
        index as input and returns a device string indicating which
        device the shard's infeed should be placed on. If placement_function
        and tpu_ordinal_function are None, inputs are sharded round-robin
        across the devices in the system.
      tpu_ordinal_function: if not None, a function that takes the
        shard index as input and returns the ordinal of the TPU device
        the shard's infeed should be placed on. If placement_function
        and tpu_ordinal_function are None, inputs are sharded round-robin
        across the devices in the system.

    Returns:
      A list of host-side Ops, one for each shard, that when executed together
      will enqueue a full-size element of infeed.

    Raises:
      ValueError: if the queue configuration has previously been frozen and the
        shapes of the elements of inputs are not compatible with the frozen
        configuration.
      TypeError: if the queue configuration has previously been frozen and the
        types of the elements of inputs are not compatible with the frozen
        configuration.
    """
    if device_assignment is None:
        if placement_function is None:
            placement_function = self._default_placement_function
        if tpu_ordinal_function is None:
            tpu_ordinal_function = self._default_ordinal_function
    else:

        def _placement_function_from_map(index):
            return device_assignment.host_device(replica=index)

        def _ordinal_function_from_map(index):
            return device_assignment.tpu_ordinal(replica=index)
        if placement_function is None:
            placement_function = _placement_function_from_map
        if tpu_ordinal_function is None:
            tpu_ordinal_function = _ordinal_function_from_map
    self.set_configuration_from_input_tensors(inputs)
    self.freeze()
    if self._generated_enqueue_ops and (not ops.inside_function()):
        raise ValueError("Can't generate two enqueue Ops from the same queue")
    self._generated_enqueue_ops = True
    split_name_prefix = '%s/split' % self._name
    if self.number_of_shards == 1:
        transposed_sharded_inputs = [[inp] for inp in inputs]
    else:

        def split_fn(inp, num_shards, axis, name):
            with ops.colocate_with(inp):
                return array_ops.split(inp, num_shards, axis=axis, name=name)
        transposed_sharded_inputs = [split_fn(inp, self.number_of_shards, axis=policy.shard_dimension, name='%s/%d' % (split_name_prefix, index)) for inp, policy, index in zip(inputs, self._sharding_policies, range(self.number_of_tuple_elements))]
    sharded_inputs = [[shard[i] for shard in transposed_sharded_inputs] for i in range(self.number_of_shards)]
    name_prefix = '%s/enqueue' % self._name
    return [self._generate_enqueue_op(shard, name_prefix, index, device=placement_function(index), tpu_ordinal=tpu_ordinal_function(index)) for shard, index in zip(sharded_inputs, range(self.number_of_shards))]
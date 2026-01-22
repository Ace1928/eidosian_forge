import copy
import threading
from typing import Callable, List, Optional, Union
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
def batch_all_reduce(self, input_tensor_packs: List[List[core.TensorLike]], options: Optional[collective_util.Options]=None) -> core.Tensor:
    """Batch all-reduce dense tensors.

    This takes a list of batches of tensors. Using multiple batches have the
    benefit that it doesn't need to wait for all inputs to be ready to start the
    all-reduce.

    Args:
      input_tensor_packs: a list of lists of dense tensors.
      options: an optional tf.distribute.experimental.CommunicationOptions. If
        provided, it overrides the default options.

    Returns:
      A flat list of reduced tensors.
    """
    options = self._options.merge(options)
    outputs = []
    for pack in input_tensor_packs:
        if context.executing_eagerly():
            for input_tensor in pack:
                outputs.append(self.all_reduce(input_tensor, None, options))
        else:
            with ops.device(self._device):
                flat_tensors = [array_ops.reshape(t, [-1]) for t in pack]
                shapes = [array_ops.shape(t) for t in pack]
                if options.implementation == collective_util.CommunicationImplementation.NCCL and outputs:
                    control_input = outputs[-1]
                else:
                    control_input = None
                reduced = self.all_reduce(array_ops.concat(flat_tensors, axis=0), control_input, options)
                num_elements = [math_ops.reduce_prod(s) for s in shapes]
                flat_outputs = array_ops.split(reduced, num_elements, axis=0)
                for shape, flat_output in zip(shapes, flat_outputs):
                    outputs.append(array_ops.reshape(flat_output, shape))
    return outputs
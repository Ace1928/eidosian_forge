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
def tag_sharding_attribute_for_dequeued_tensors(dequeues, dims):
    """Tags appropriate XLA sharding attribute to the dequeued tensors.

  Args:
    dequeues: A list of dequeued tensors on TPU.
    dims: A list of integer describes how the tensor is partitioned.

  Returns:
    The same dequeues with appropriate xla_sharding attribute.
  """
    nest.assert_shallow_structure(dequeues, dims)
    return nest.map_structure_up_to(dequeues, _tag_sharding_attribute_for_dequeued_tensor, dequeues, dims)
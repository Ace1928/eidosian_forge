from enum import Enum
import functools
import weakref
import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as parallel_control_flow_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import tf_decorator
def ragged_assert_compatible_and_get_flat_values(values, mask=None):
    """If ragged, it checks the compatibility and then returns the flat_values.

     Note: If two tensors are dense, it does not check their compatibility.
     Note: Although two ragged tensors with different ragged ranks could have
           identical overall rank and dimension sizes and hence be compatible,
           we do not support those cases.
  Args:
     values: A list of potentially ragged tensor of the same ragged_rank.
     mask: A potentially ragged tensor of the same ragged_rank as elements in
       Values.

  Returns:
     A tuple in which the first element is the list of tensors and the second
     is the mask tensor. ([Values], mask). Mask and the element in Values
     are equal to the flat_values of the input arguments (if they were ragged).
  """
    if isinstance(values, list):
        is_all_ragged = all((isinstance(rt, ragged_tensor.RaggedTensor) for rt in values))
        is_any_ragged = any((isinstance(rt, ragged_tensor.RaggedTensor) for rt in values))
    else:
        is_all_ragged = isinstance(values, ragged_tensor.RaggedTensor)
        is_any_ragged = is_all_ragged
    if is_all_ragged and (mask is None or isinstance(mask, ragged_tensor.RaggedTensor)):
        to_be_stripped = False
        if not isinstance(values, list):
            values = [values]
            to_be_stripped = True
        nested_row_split_list = [rt.nested_row_splits for rt in values]
        assertion_list = _assert_splits_match(nested_row_split_list)
        if isinstance(mask, ragged_tensor.RaggedTensor):
            assertion_list_for_mask = _assert_splits_match([nested_row_split_list[0], mask.nested_row_splits])
            with ops.control_dependencies(assertion_list_for_mask):
                mask = array_ops.expand_dims(mask.flat_values, -1)
        flat_values = []
        for value in values:
            with ops.control_dependencies(assertion_list):
                flat_values.append(array_ops.expand_dims(value.flat_values, -1))
        values = flat_values[0] if to_be_stripped else flat_values
    elif is_any_ragged:
        raise TypeError('One of the inputs does not have acceptable types.')
    elif isinstance(mask, ragged_tensor.RaggedTensor):
        raise TypeError('Ragged mask is not allowed with non-ragged inputs.')
    return (values, mask)
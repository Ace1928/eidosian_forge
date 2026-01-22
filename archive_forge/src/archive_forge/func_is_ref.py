import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.util import nest
def is_ref(x):
    """Evaluates if the object has reference semantics.

  An object is deemed "reference" if it is a `tf.Variable` instance or is
  derived from a `tf.Module` with `dtype` and `shape` properties.

  Args:
    x: Any object.

  Returns:
    is_ref: Python `bool` indicating input is has nonreference semantics, i.e.,
      is a `tf.Variable` or a `tf.Module` with `dtype` and `shape` properties.
  """
    return isinstance(x, variables_module.Variable) or (isinstance(x, module.Module) and hasattr(x, 'dtype') and hasattr(x, 'shape'))
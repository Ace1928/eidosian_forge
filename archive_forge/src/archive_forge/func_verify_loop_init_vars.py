import functools
import sys
import traceback
import numpy as np
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.operators import variables
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.autograph.utils import misc
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.types import distribute
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
def verify_loop_init_vars(init_vars, symbol_names, first_iter_vars=None, extra_message=None):
    """Ensures that all values in the state are valid to use in a TF loop.

  The init_vars may contain placeholder values derived from first_iter_vars.

  Args:
    init_vars: initial loop variables (as taken before entering the loop)
    symbol_names: corresponding names of the initial loop variables
    first_iter_vars: loop variables after one iteration of the loop
    extra_message: an extra string to append to the error message, in case of
      "undefined variable" errors (see variables.Undefined)
  """
    if not symbol_names:
        return
    if first_iter_vars is None:
        first_iter_vars = (None,) * len(symbol_names)
    assert len(symbol_names) == len(init_vars)
    assert len(symbol_names) == len(first_iter_vars)
    for name, val, fi_val in zip(symbol_names, init_vars, first_iter_vars):
        if isinstance(val, variables.UndefinedReturnValue):
            if fi_val:
                raise ValueError('the return value from a TensorFlow loop may only be a {}; got {}'.format(LEGAL_LOOP_TYPES, type(fi_val)))
            else:
                raise NotImplementedError('a return statement cannot be placed inside this TensorFlow loop; this may happen if a return statement depends on a static Python condition such as a hyperparameter')
        error_msg = None
        if val is None:
            error_msg = "'{}' is not allowed to be None before the loop".format(name)
        elif isinstance(val, variables.Undefined):
            error_msg = "'{}' must be defined before the loop".format(name)
            if extra_message:
                error_msg += '\n' + extra_message
        if error_msg is not None:
            raise ValueError(error_msg)
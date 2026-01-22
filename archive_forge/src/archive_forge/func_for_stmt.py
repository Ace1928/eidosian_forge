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
def for_stmt(iter_, extra_test, body, get_state, set_state, symbol_names, opts):
    """Functional form of a for statement.

  The loop operates on a state, which includes all symbols that are
  variant across loop iterations, excluding the variables local to the loop.

  For example, given the loop below that calculates the geometric and
  arithmetic means or some numbers:

  ```
    geo_mean = 1
    arith_mean = 0
    for i in range(n):
      a = numbers[i]
      geo_mean *= a
      arith_mean += a
  ```

  The state is represented by the variables named geo_mean and arith_mean. The
  `extra_test`, `body`, `get_state` and `set_state` functions must bind to the
  original `geo_mean` and `arith_mean` symbols, using `nonlocal`.

  The inputs and outputs of the callables representing the loop blocks are not
  explicit - instead, these functions must use nonlocal/global for side effects.
  The inputs and outputs are instead controlled by the set_state/get_state
  functions.

  Args:
    iter_: The entity being iterated over.
    extra_test: Callable with boolean return type. An additional loop condition.
    body: Callable representing the actual loop body.
    get_state: Additional callable which can capture additional state (such as
      the values of composite symbols). This is only useful when staging the
      loop.
    set_state: Additional callable which save values captured by get_state back
      into the Python environment. This is only useful when staging the loop.
    symbol_names: Tuple containing names of the loop variables returned by
      get_state.
    opts: Optional dict of extra loop parameters.
  """
    try:
        for_fn = for_loop_registry.lookup(iter_)
    except LookupError:
        for_fn = _py_for_stmt
    if tensor_util.is_tf_type(iter_):
        if tensors.is_range_tensor(iter_):
            for_fn = _tf_range_for_stmt
        elif isinstance(iter_, ragged_tensor.RaggedTensor):
            for_fn = _tf_ragged_for_stmt
        else:
            for_fn = _known_len_tf_for_stmt
    elif isinstance(iter_, distribute.Iterator):
        for_fn = _tf_iterator_for_stmt
    elif isinstance(iter_, distribute.Iterable):
        for_fn = _tf_distributed_iterable_for_stmt
    for_fn(iter_, extra_test, body, get_state, set_state, symbol_names, opts)
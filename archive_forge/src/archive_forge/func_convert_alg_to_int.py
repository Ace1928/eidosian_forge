import enum
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
def convert_alg_to_int(alg):
    """Converts algorithm to an integer.

  Args:
    alg: can be one of these types: integer, Algorithm, Tensor, string. Allowed
      strings are "philox" and "threefry".

  Returns:
    An integer, unless the input is a Tensor in which case a Tensor is returned.
  """
    if isinstance(alg, int):
        return alg
    if isinstance(alg, Algorithm):
        return alg.value
    if isinstance(alg, tensor.Tensor):
        return alg
    if isinstance(alg, str):
        canon_alg = alg.strip().lower().replace('-', '').replace('_', '')
        if canon_alg == 'philox':
            return Algorithm.PHILOX.value
        elif canon_alg == 'threefry':
            return Algorithm.THREEFRY.value
        elif canon_alg == 'autoselect':
            return Algorithm.AUTO_SELECT.value
        else:
            raise ValueError(unsupported_alg_error_msg(alg))
    else:
        raise TypeError(f"Can't convert argument `alg` (of value {alg} and type {type(alg)}) to int.")
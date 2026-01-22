import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
Splits input tensor across all dimensions.

  An op which slices the input tensor based on the given num_splits attribute,
  pads slices optionally, and returned the slices. Slices are returned in
  row-major order.

  This op may be generated via the TPU bridge.

  For example, with `input` tensor:
  ```
  [[0, 1, 2],
   [3, 4, 5],
   [6, 7, 8]]
  ```
  `num_splits`:
  ```
  [2, 2]
  ```
  and `paddings`:
  ```
  [1, 1]
  ```
  the expected `outputs` is:
  ```
  [[0, 1],
   [3, 4]]
  [[2, 0],
   [5, 0]]
  [[6, 7],
   [0, 0]]
  [[8, 0],
   [0, 0]]
  ```

  Args:
    input: A `Tensor`. Input tensor to split across all dimensions.
        }
        out_arg {
          name: "outputs"
          description: <<END
      Output slices based on input and num_splits defined, in row-major order.
    N: An `int` that is `>= 1`.
    num_splits: A list of `ints`.
      Number of ways to split per dimension. Shape dimensions must be evenly
      divisible.
    paddings: An optional list of `ints`. Defaults to `[]`.
      Optional list of right paddings per dimension of input tensor to apply before
      splitting. This can be used to make a dimension evenly divisible.
    name: A name for the operation (optional).

  Returns:
    A list of `N` `Tensor` objects with the same type as `input`.
  
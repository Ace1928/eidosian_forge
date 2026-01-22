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
def xla_split_nd(input: _atypes.TensorFuzzingAnnotation[TV_XlaSplitND_T], N: int, num_splits, paddings=[], name=None):
    """Splits input tensor across all dimensions.

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
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaSplitND', name, input, 'N', N, 'num_splits', num_splits, 'paddings', paddings)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return xla_split_nd_eager_fallback(input, N=N, num_splits=num_splits, paddings=paddings, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    N = _execute.make_int(N, 'N')
    if not isinstance(num_splits, (list, tuple)):
        raise TypeError("Expected list for 'num_splits' argument to 'xla_split_nd' Op, not %r." % num_splits)
    num_splits = [_execute.make_int(_i, 'num_splits') for _i in num_splits]
    if paddings is None:
        paddings = []
    if not isinstance(paddings, (list, tuple)):
        raise TypeError("Expected list for 'paddings' argument to 'xla_split_nd' Op, not %r." % paddings)
    paddings = [_execute.make_int(_i, 'paddings') for _i in paddings]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaSplitND', input=input, N=N, num_splits=num_splits, paddings=paddings, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'N', _op._get_attr_int('N'), 'num_splits', _op.get_attr('num_splits'), 'paddings', _op.get_attr('paddings'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaSplitND', _inputs_flat, _attrs, _result)
    return _result
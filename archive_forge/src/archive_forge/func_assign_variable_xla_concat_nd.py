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
def assign_variable_xla_concat_nd(resource: _atypes.TensorFuzzingAnnotation[_atypes.Resource], inputs: List[_atypes.TensorFuzzingAnnotation[TV_AssignVariableXlaConcatND_T]], num_concats, paddings=[], name=None):
    """Concats input tensor across all dimensions.

  An op which merges slices the input tensor based on the given num_splits
  attribute, strips paddings optionally, and writes the merged tensor without
  paddings to the resource variable.

  This op may be generated via the TPU bridge.

  For example, with `input` tensor:
  ```
  [[0, 1],
   [4, 5]]
  [[2, 3],
   [6, 7]]
  [[8, 9],
   [12, 13]]
  [[10, 11],
   [14, 15]]
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
  [[0, 1, 2],
   [4, 5, 6],
   [8, 9, 10]]
  ```

  Args:
    resource: A `Tensor` of type `resource`.
      Resource variable for concatenated input tensors across all dimensions.
        }
        in_arg {
          name: "inputs"
          description: <<END
      Input tensor slices in row-major order to merge across all dimensions. All
      inputs must have the same shape.
        }
        out_arg {
          name: "output"
          description: <<END
      Output tensor formed from merging input slices based on num_concats defined.
    inputs: A list of at least 1 `Tensor` objects with the same type.
    num_concats: A list of `ints`. Number of ways to merge per dimension.
    paddings: An optional list of `ints`. Defaults to `[]`.
      Optional list of right paddings per dimension to strip from the final merged
      tensor. These paddings must not exceed the dimension size of the merged result
      prior to stripping paddings.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'AssignVariableXlaConcatND', name, resource, inputs, 'num_concats', num_concats, 'paddings', paddings)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return assign_variable_xla_concat_nd_eager_fallback(resource, inputs, num_concats=num_concats, paddings=paddings, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(inputs, (list, tuple)):
        raise TypeError("Expected list for 'inputs' argument to 'assign_variable_xla_concat_nd' Op, not %r." % inputs)
    _attr_N = len(inputs)
    if not isinstance(num_concats, (list, tuple)):
        raise TypeError("Expected list for 'num_concats' argument to 'assign_variable_xla_concat_nd' Op, not %r." % num_concats)
    num_concats = [_execute.make_int(_i, 'num_concats') for _i in num_concats]
    if paddings is None:
        paddings = []
    if not isinstance(paddings, (list, tuple)):
        raise TypeError("Expected list for 'paddings' argument to 'assign_variable_xla_concat_nd' Op, not %r." % paddings)
    paddings = [_execute.make_int(_i, 'paddings') for _i in paddings]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('AssignVariableXlaConcatND', resource=resource, inputs=inputs, num_concats=num_concats, paddings=paddings, name=name)
    return _op
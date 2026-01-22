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
def mirror_pad(input: _atypes.TensorFuzzingAnnotation[TV_MirrorPad_T], paddings: _atypes.TensorFuzzingAnnotation[TV_MirrorPad_Tpaddings], mode: str, name=None) -> _atypes.TensorFuzzingAnnotation[TV_MirrorPad_T]:
    """Pads a tensor with mirrored values.

  This operation pads a `input` with mirrored values according to the `paddings`
  you specify. `paddings` is an integer tensor with shape `[n, 2]`, where n is
  the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
  how many values to add before the contents of `input` in that dimension, and
  `paddings[D, 1]` indicates how many values to add after the contents of `input`
  in that dimension. Both `paddings[D, 0]` and `paddings[D, 1]` must be no greater
  than `input.dim_size(D)` (or `input.dim_size(D) - 1`) if `copy_border` is true
  (if false, respectively).

  The padded size of each dimension D of the output is:

  `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

  For example:

  ```
  # 't' is [[1, 2, 3], [4, 5, 6]].
  # 'paddings' is [[1, 1]], [2, 2]].
  # 'mode' is SYMMETRIC.
  # rank of 't' is 2.
  pad(t, paddings) ==> [[2, 1, 1, 2, 3, 3, 2]
                        [2, 1, 1, 2, 3, 3, 2]
                        [5, 4, 4, 5, 6, 6, 5]
                        [5, 4, 4, 5, 6, 6, 5]]
  ```

  Args:
    input: A `Tensor`. The input tensor to be padded.
    paddings: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A two-column matrix specifying the padding sizes. The number of
      rows must be the same as the rank of `input`.
    mode: A `string` from: `"REFLECT", "SYMMETRIC"`.
      Either `REFLECT` or `SYMMETRIC`. In reflect mode the padded regions
      do not include the borders, while in symmetric mode the padded regions
      do include the borders. For example, if `input` is `[1, 2, 3]` and `paddings`
      is `[0, 2]`, then the output is `[1, 2, 3, 2, 1]` in reflect mode, and
      it is `[1, 2, 3, 3, 2]` in symmetric mode.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MirrorPad', name, input, paddings, 'mode', mode)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return mirror_pad_eager_fallback(input, paddings, mode=mode, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    mode = _execute.make_str(mode, 'mode')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('MirrorPad', input=input, paddings=paddings, mode=mode, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tpaddings', _op._get_attr_type('Tpaddings'), 'mode', _op.get_attr('mode'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('MirrorPad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
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
def save_v2(prefix: _atypes.TensorFuzzingAnnotation[_atypes.String], tensor_names: _atypes.TensorFuzzingAnnotation[_atypes.String], shape_and_slices: _atypes.TensorFuzzingAnnotation[_atypes.String], tensors, name=None):
    """Saves tensors in V2 checkpoint format.

  By default, saves the named tensors in full.  If the caller wishes to save
  specific slices of full tensors, "shape_and_slices" should be non-empty strings
  and correspondingly well-formed.

  Args:
    prefix: A `Tensor` of type `string`.
      Must have a single element. The prefix of the V2 checkpoint to which we
      write the tensors.
    tensor_names: A `Tensor` of type `string`.
      shape {N}. The names of the tensors to be saved.
    shape_and_slices: A `Tensor` of type `string`.
      shape {N}.  The slice specs of the tensors to be saved.
      Empty strings indicate that they are non-partitioned tensors.
    tensors: A list of `Tensor` objects. `N` tensors to save.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SaveV2', name, prefix, tensor_names, shape_and_slices, tensors)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return save_v2_eager_fallback(prefix, tensor_names, shape_and_slices, tensors, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SaveV2', prefix=prefix, tensor_names=tensor_names, shape_and_slices=shape_and_slices, tensors=tensors, name=name)
    return _op
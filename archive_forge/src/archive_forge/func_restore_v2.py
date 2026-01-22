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
def restore_v2(prefix: _atypes.TensorFuzzingAnnotation[_atypes.String], tensor_names: _atypes.TensorFuzzingAnnotation[_atypes.String], shape_and_slices: _atypes.TensorFuzzingAnnotation[_atypes.String], dtypes, name=None):
    """Restores tensors from a V2 checkpoint.

  For backward compatibility with the V1 format, this Op currently allows
  restoring from a V1 checkpoint as well:
    - This Op first attempts to find the V2 index file pointed to by "prefix", and
      if found proceed to read it as a V2 checkpoint;
    - Otherwise the V1 read path is invoked.
  Relying on this behavior is not recommended, as the ability to fall back to read
  V1 might be deprecated and eventually removed.

  By default, restores the named tensors in full.  If the caller wishes to restore
  specific slices of stored tensors, "shape_and_slices" should be non-empty
  strings and correspondingly well-formed.

  Callers must ensure all the named tensors are indeed stored in the checkpoint.

  Args:
    prefix: A `Tensor` of type `string`.
      Must have a single element.  The prefix of a V2 checkpoint.
    tensor_names: A `Tensor` of type `string`.
      shape {N}.  The names of the tensors to be restored.
    shape_and_slices: A `Tensor` of type `string`.
      shape {N}.  The slice specs of the tensors to be restored.
      Empty strings indicate that they are non-partitioned tensors.
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
      shape {N}.  The list of expected dtype for the tensors.  Must match
      those stored in the checkpoint.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RestoreV2', name, prefix, tensor_names, shape_and_slices, 'dtypes', dtypes)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return restore_v2_eager_fallback(prefix, tensor_names, shape_and_slices, dtypes=dtypes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(dtypes, (list, tuple)):
        raise TypeError("Expected list for 'dtypes' argument to 'restore_v2' Op, not %r." % dtypes)
    dtypes = [_execute.make_type(_t, 'dtypes') for _t in dtypes]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RestoreV2', prefix=prefix, tensor_names=tensor_names, shape_and_slices=shape_and_slices, dtypes=dtypes, name=name)
    _result = _outputs[:]
    if not _result:
        return _op
    if _execute.must_record_gradient():
        _attrs = ('dtypes', _op.get_attr('dtypes'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RestoreV2', _inputs_flat, _attrs, _result)
    return _result
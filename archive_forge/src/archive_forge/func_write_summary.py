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
def write_summary(writer: _atypes.TensorFuzzingAnnotation[_atypes.Resource], step: _atypes.TensorFuzzingAnnotation[_atypes.Int64], tensor: _atypes.TensorFuzzingAnnotation[TV_WriteSummary_T], tag: _atypes.TensorFuzzingAnnotation[_atypes.String], summary_metadata: _atypes.TensorFuzzingAnnotation[_atypes.String], name=None):
    """Writes a tensor summary.

  Writes `tensor` at `step` with `tag` using summary `writer`.

  Args:
    writer: A `Tensor` of type `resource`.
    step: A `Tensor` of type `int64`.
    tensor: A `Tensor`.
    tag: A `Tensor` of type `string`.
    summary_metadata: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'WriteSummary', name, writer, step, tensor, tag, summary_metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return write_summary_eager_fallback(writer, step, tensor, tag, summary_metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('WriteSummary', writer=writer, step=step, tensor=tensor, tag=tag, summary_metadata=summary_metadata, name=name)
    return _op
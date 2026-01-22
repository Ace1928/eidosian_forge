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
def write_audio_summary(writer: _atypes.TensorFuzzingAnnotation[_atypes.Resource], step: _atypes.TensorFuzzingAnnotation[_atypes.Int64], tag: _atypes.TensorFuzzingAnnotation[_atypes.String], tensor: _atypes.TensorFuzzingAnnotation[_atypes.Float32], sample_rate: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_outputs: int=3, name=None):
    """Writes an audio summary.

  Writes encoded audio summary `tensor` at `step` with `tag` using summary `writer`.
  `sample_rate` is the audio sample rate is Hz.

  Args:
    writer: A `Tensor` of type `resource`.
    step: A `Tensor` of type `int64`.
    tag: A `Tensor` of type `string`.
    tensor: A `Tensor` of type `float32`.
    sample_rate: A `Tensor` of type `float32`.
    max_outputs: An optional `int` that is `>= 1`. Defaults to `3`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'WriteAudioSummary', name, writer, step, tag, tensor, sample_rate, 'max_outputs', max_outputs)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return write_audio_summary_eager_fallback(writer, step, tag, tensor, sample_rate, max_outputs=max_outputs, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if max_outputs is None:
        max_outputs = 3
    max_outputs = _execute.make_int(max_outputs, 'max_outputs')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('WriteAudioSummary', writer=writer, step=step, tag=tag, tensor=tensor, sample_rate=sample_rate, max_outputs=max_outputs, name=name)
    return _op
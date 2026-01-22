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
def fixed_length_record_dataset_v2(filenames: _atypes.TensorFuzzingAnnotation[_atypes.String], header_bytes: _atypes.TensorFuzzingAnnotation[_atypes.Int64], record_bytes: _atypes.TensorFuzzingAnnotation[_atypes.Int64], footer_bytes: _atypes.TensorFuzzingAnnotation[_atypes.Int64], buffer_size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], compression_type: _atypes.TensorFuzzingAnnotation[_atypes.String], metadata: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """TODO: add doc.

  Args:
    filenames: A `Tensor` of type `string`.
    header_bytes: A `Tensor` of type `int64`.
    record_bytes: A `Tensor` of type `int64`.
    footer_bytes: A `Tensor` of type `int64`.
    buffer_size: A `Tensor` of type `int64`.
    compression_type: A `Tensor` of type `string`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'FixedLengthRecordDatasetV2', name, filenames, header_bytes, record_bytes, footer_bytes, buffer_size, compression_type, 'metadata', metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return fixed_length_record_dataset_v2_eager_fallback(filenames, header_bytes, record_bytes, footer_bytes, buffer_size, compression_type, metadata=metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('FixedLengthRecordDatasetV2', filenames=filenames, header_bytes=header_bytes, record_bytes=record_bytes, footer_bytes=footer_bytes, buffer_size=buffer_size, compression_type=compression_type, metadata=metadata, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('metadata', _op.get_attr('metadata'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('FixedLengthRecordDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
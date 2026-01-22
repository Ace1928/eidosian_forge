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
def reader_read_up_to(reader_handle: _atypes.TensorFuzzingAnnotation[_atypes.String], queue_handle: _atypes.TensorFuzzingAnnotation[_atypes.String], num_records: _atypes.TensorFuzzingAnnotation[_atypes.Int64], name=None):
    """Returns up to `num_records` (key, value) pairs produced by a Reader.

  Will dequeue from the input queue if necessary (e.g. when the
  Reader needs to start reading from a new file since it has finished
  with the previous file).
  It may return less than `num_records` even before the last batch.

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a `Reader`.
    queue_handle: A `Tensor` of type mutable `string`.
      Handle to a `Queue`, with string work items.
    num_records: A `Tensor` of type `int64`.
      number of records to read from `Reader`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (keys, values).

    keys: A `Tensor` of type `string`.
    values: A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("reader_read_up_to op does not support eager execution. Arg 'queue_handle' is a ref.")
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ReaderReadUpTo', reader_handle=reader_handle, queue_handle=queue_handle, num_records=num_records, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ()
        _inputs_flat = _op.inputs
        _execute.record_gradient('ReaderReadUpTo', _inputs_flat, _attrs, _result)
    _result = _ReaderReadUpToOutput._make(_result)
    return _result
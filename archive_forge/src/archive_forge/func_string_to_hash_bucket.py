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
def string_to_hash_bucket(string_tensor: _atypes.TensorFuzzingAnnotation[_atypes.String], num_buckets: int, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Int64]:
    """Converts each string in the input Tensor to its hash mod by a number of buckets.

  The hash function is deterministic on the content of the string within the
  process.

  Note that the hash function may change from time to time.
  This functionality will be deprecated and it's recommended to use
  `tf.string_to_hash_bucket_fast()` or `tf.string_to_hash_bucket_strong()`.

  Args:
    string_tensor: A `Tensor` of type `string`.
    num_buckets: An `int` that is `>= 1`. The number of buckets.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StringToHashBucket', name, string_tensor, 'num_buckets', num_buckets)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return string_to_hash_bucket_eager_fallback(string_tensor, num_buckets=num_buckets, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    num_buckets = _execute.make_int(num_buckets, 'num_buckets')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StringToHashBucket', string_tensor=string_tensor, num_buckets=num_buckets, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('num_buckets', _op._get_attr_int('num_buckets'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StringToHashBucket', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('strings.to_hash_bucket_fast', v1=['strings.to_hash_bucket_fast', 'string_to_hash_bucket_fast'])
@deprecated_endpoints('string_to_hash_bucket_fast')
def string_to_hash_bucket_fast(input: _atypes.TensorFuzzingAnnotation[_atypes.String], num_buckets: int, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Int64]:
    """Converts each string in the input Tensor to its hash mod by a number of buckets.

  The hash function is deterministic on the content of the string within the
  process and will never change. However, it is not suitable for cryptography.
  This function may be used when CPU time is scarce and inputs are trusted or
  unimportant. There is a risk of adversaries constructing inputs that all hash
  to the same bucket. To prevent this problem, use a strong hash function with
  `tf.string_to_hash_bucket_strong`.

  Examples:

  >>> tf.strings.to_hash_bucket_fast(["Hello", "TensorFlow", "2.x"], 3).numpy()
  array([0, 2, 2])

  Args:
    input: A `Tensor` of type `string`. The strings to assign a hash bucket.
    num_buckets: An `int` that is `>= 1`. The number of buckets.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StringToHashBucketFast', name, input, 'num_buckets', num_buckets)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_string_to_hash_bucket_fast((input, num_buckets, name), None)
            if _result is not NotImplemented:
                return _result
            return string_to_hash_bucket_fast_eager_fallback(input, num_buckets=num_buckets, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(string_to_hash_bucket_fast, (), dict(input=input, num_buckets=num_buckets, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_string_to_hash_bucket_fast((input, num_buckets, name), None)
        if _result is not NotImplemented:
            return _result
    num_buckets = _execute.make_int(num_buckets, 'num_buckets')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('StringToHashBucketFast', input=input, num_buckets=num_buckets, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(string_to_hash_bucket_fast, (), dict(input=input, num_buckets=num_buckets, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('num_buckets', _op._get_attr_int('num_buckets'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StringToHashBucketFast', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
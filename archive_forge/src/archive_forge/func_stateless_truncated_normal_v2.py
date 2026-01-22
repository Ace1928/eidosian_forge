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
def stateless_truncated_normal_v2(shape: _atypes.TensorFuzzingAnnotation[TV_StatelessTruncatedNormalV2_Tshape], key: _atypes.TensorFuzzingAnnotation[_atypes.UInt64], counter: _atypes.TensorFuzzingAnnotation[_atypes.UInt64], alg: _atypes.TensorFuzzingAnnotation[_atypes.Int32], dtype: TV_StatelessTruncatedNormalV2_dtype=_dtypes.float32, name=None) -> _atypes.TensorFuzzingAnnotation[TV_StatelessTruncatedNormalV2_dtype]:
    """Outputs deterministic pseudorandom values from a truncated normal distribution.

  The generated values follow a normal distribution with mean 0 and standard
  deviation 1, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.

  The outputs are a deterministic function of `shape`, `key`, `counter` and `alg`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    key: A `Tensor` of type `uint64`.
      Key for the counter-based RNG algorithm (shape uint64[1]).
    counter: A `Tensor` of type `uint64`.
      Initial counter for the counter-based RNG algorithm (shape uint64[2] or uint64[1] depending on the algorithm). If a larger vector is given, only the needed portion on the left (i.e. [:N]) will be used.
    alg: A `Tensor` of type `int32`. The RNG algorithm (shape int32[]).
    dtype: An optional `tf.DType` from: `tf.half, tf.bfloat16, tf.float32, tf.float64`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StatelessTruncatedNormalV2', name, shape, key, counter, alg, 'dtype', dtype)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return stateless_truncated_normal_v2_eager_fallback(shape, key, counter, alg, dtype=dtype, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if dtype is None:
        dtype = _dtypes.float32
    dtype = _execute.make_type(dtype, 'dtype')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StatelessTruncatedNormalV2', shape=shape, key=key, counter=counter, alg=alg, dtype=dtype, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'Tshape', _op._get_attr_type('Tshape'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StatelessTruncatedNormalV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
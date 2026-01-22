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
def stateful_random_binomial(resource: _atypes.TensorFuzzingAnnotation[_atypes.Resource], algorithm: _atypes.TensorFuzzingAnnotation[_atypes.Int64], shape: _atypes.TensorFuzzingAnnotation[TV_StatefulRandomBinomial_S], counts: _atypes.TensorFuzzingAnnotation[TV_StatefulRandomBinomial_T], probs: _atypes.TensorFuzzingAnnotation[TV_StatefulRandomBinomial_T], dtype: TV_StatefulRandomBinomial_dtype=_dtypes.int64, name=None) -> _atypes.TensorFuzzingAnnotation[TV_StatefulRandomBinomial_dtype]:
    """TODO: add doc.

  Args:
    resource: A `Tensor` of type `resource`.
    algorithm: A `Tensor` of type `int64`.
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    counts: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
    probs: A `Tensor`. Must have the same type as `counts`.
    dtype: An optional `tf.DType` from: `tf.half, tf.float32, tf.float64, tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StatefulRandomBinomial', name, resource, algorithm, shape, counts, probs, 'dtype', dtype)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return stateful_random_binomial_eager_fallback(resource, algorithm, shape, counts, probs, dtype=dtype, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if dtype is None:
        dtype = _dtypes.int64
    dtype = _execute.make_type(dtype, 'dtype')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StatefulRandomBinomial', resource=resource, algorithm=algorithm, shape=shape, counts=counts, probs=probs, dtype=dtype, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('S', _op._get_attr_type('S'), 'T', _op._get_attr_type('T'), 'dtype', _op._get_attr_type('dtype'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StatefulRandomBinomial', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
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
def multinomial(logits: _atypes.TensorFuzzingAnnotation[TV_Multinomial_T], num_samples: _atypes.TensorFuzzingAnnotation[_atypes.Int32], seed: int=0, seed2: int=0, output_dtype: TV_Multinomial_output_dtype=_dtypes.int64, name=None) -> _atypes.TensorFuzzingAnnotation[TV_Multinomial_output_dtype]:
    """Draws samples from a multinomial distribution.

  Args:
    logits: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i, :]`
      represents the unnormalized log probabilities for all classes.
    num_samples: A `Tensor` of type `int32`.
      0-D.  Number of independent samples to draw for each row slice.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 is set to be non-zero, the internal random number
      generator is seeded by the given seed.  Otherwise, a random seed is used.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    output_dtype: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_dtype`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Multinomial', name, logits, num_samples, 'seed', seed, 'seed2', seed2, 'output_dtype', output_dtype)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return multinomial_eager_fallback(logits, num_samples, seed=seed, seed2=seed2, output_dtype=output_dtype, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if seed is None:
        seed = 0
    seed = _execute.make_int(seed, 'seed')
    if seed2 is None:
        seed2 = 0
    seed2 = _execute.make_int(seed2, 'seed2')
    if output_dtype is None:
        output_dtype = _dtypes.int64
    output_dtype = _execute.make_type(output_dtype, 'output_dtype')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('Multinomial', logits=logits, num_samples=num_samples, seed=seed, seed2=seed2, output_dtype=output_dtype, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('seed', _op._get_attr_int('seed'), 'seed2', _op._get_attr_int('seed2'), 'T', _op._get_attr_type('T'), 'output_dtype', _op._get_attr_type('output_dtype'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('Multinomial', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
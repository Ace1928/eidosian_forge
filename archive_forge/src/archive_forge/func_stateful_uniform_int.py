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
def stateful_uniform_int(resource: _atypes.TensorFuzzingAnnotation[_atypes.Resource], algorithm: _atypes.TensorFuzzingAnnotation[_atypes.Int64], shape: _atypes.TensorFuzzingAnnotation[TV_StatefulUniformInt_shape_dtype], minval: _atypes.TensorFuzzingAnnotation[TV_StatefulUniformInt_dtype], maxval: _atypes.TensorFuzzingAnnotation[TV_StatefulUniformInt_dtype], name=None) -> _atypes.TensorFuzzingAnnotation[TV_StatefulUniformInt_dtype]:
    """Outputs random integers from a uniform distribution.

  The generated values are uniform integers in the range `[minval, maxval)`.
  The lower bound `minval` is included in the range, while the upper bound
  `maxval` is excluded.

  The random integers are slightly biased unless `maxval - minval` is an exact
  power of two.  The bias is small for values of `maxval - minval` significantly
  smaller than the range of the output (either `2^32` or `2^64`).

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    minval: A `Tensor`. Minimum value (inclusive, scalar).
    maxval: A `Tensor`. Must have the same type as `minval`.
      Maximum value (exclusive, scalar).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `minval`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StatefulUniformInt', name, resource, algorithm, shape, minval, maxval)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return stateful_uniform_int_eager_fallback(resource, algorithm, shape, minval, maxval, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StatefulUniformInt', resource=resource, algorithm=algorithm, shape=shape, minval=minval, maxval=maxval, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'shape_dtype', _op._get_attr_type('shape_dtype'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StatefulUniformInt', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
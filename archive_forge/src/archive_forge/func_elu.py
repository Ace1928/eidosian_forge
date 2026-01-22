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
@tf_export('nn.elu')
def elu(features: _atypes.TensorFuzzingAnnotation[TV_Elu_T], name=None) -> _atypes.TensorFuzzingAnnotation[TV_Elu_T]:
    """Computes the exponential linear function.

  The ELU function is defined as:

   * $ e ^ x - 1 $ if $ x < 0 $
   * $ x $ if $ x >= 0 $

  Examples:

  >>> tf.nn.elu(1.0)
  <tf.Tensor: shape=(), dtype=float32, numpy=1.0>
  >>> tf.nn.elu(0.0)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.0>
  >>> tf.nn.elu(-1000.0)
  <tf.Tensor: shape=(), dtype=float32, numpy=-1.0>

  See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
  ](http://arxiv.org/abs/1511.07289)

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Elu', name, features)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_elu((features, name), None)
            if _result is not NotImplemented:
                return _result
            return elu_eager_fallback(features, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(elu, (), dict(features=features, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_elu((features, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('Elu', features=features, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(elu, (), dict(features=features, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('Elu', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
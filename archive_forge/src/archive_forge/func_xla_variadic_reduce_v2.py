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
@tf_export('xla_variadic_reduce_v2')
def xla_variadic_reduce_v2(inputs, init_values, dimensions_to_reduce, reducer, name=None):
    """Wraps the variadic XLA Reduce operator.

  Semantics are documented at
   https://www.tensorflow.org/performance/xla/operation_semantics#variadic_reduce.

  This is an expanded version of XlaVariadicReduce, with support for
  operands of different dtypes, and improved shape inference.

  Args:
    inputs: A list of `Tensor` objects. the input tensor(s)
    init_values: A list of `Tensor` objects. Must have the same type as `inputs`.
      scalar initial value(s) for the reduction
    dimensions_to_reduce: A list of `ints`.
      dimension numbers over which to reduce
    reducer: A function decorated with @Defun. a reducer function to apply
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `inputs`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaVariadicReduceV2', name, inputs, init_values, 'dimensions_to_reduce', dimensions_to_reduce, 'reducer', reducer)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_variadic_reduce_v2((inputs, init_values, dimensions_to_reduce, reducer, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_variadic_reduce_v2_eager_fallback(inputs, init_values, dimensions_to_reduce=dimensions_to_reduce, reducer=reducer, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_variadic_reduce_v2, (), dict(inputs=inputs, init_values=init_values, dimensions_to_reduce=dimensions_to_reduce, reducer=reducer, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_variadic_reduce_v2((inputs, init_values, dimensions_to_reduce, reducer, name), None)
        if _result is not NotImplemented:
            return _result
    if not isinstance(dimensions_to_reduce, (list, tuple)):
        raise TypeError("Expected list for 'dimensions_to_reduce' argument to 'xla_variadic_reduce_v2' Op, not %r." % dimensions_to_reduce)
    dimensions_to_reduce = [_execute.make_int(_i, 'dimensions_to_reduce') for _i in dimensions_to_reduce]
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaVariadicReduceV2', inputs=inputs, init_values=init_values, dimensions_to_reduce=dimensions_to_reduce, reducer=reducer, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_variadic_reduce_v2, (), dict(inputs=inputs, init_values=init_values, dimensions_to_reduce=dimensions_to_reduce, reducer=reducer, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op.get_attr('T'), 'dimensions_to_reduce', _op.get_attr('dimensions_to_reduce'), 'reducer', _op.get_attr('reducer'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaVariadicReduceV2', _inputs_flat, _attrs, _result)
    return _result
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
@tf_export('xla_sharding')
def xla_sharding(input: _atypes.TensorFuzzingAnnotation[TV_XlaSharding_T], sharding: str='', unspecified_dims=[], name=None) -> _atypes.TensorFuzzingAnnotation[TV_XlaSharding_T]:
    """An op which shards the input based on the given sharding attribute. It can

  selectively annotate a subset of tensor dimensions by skipping unspecified_dims,
  and the sharding annotation should be replicated in those dims.

  Args:
    input: A `Tensor`.
    sharding: An optional `string`. Defaults to `""`.
    unspecified_dims: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaSharding', name, input, 'sharding', sharding, 'unspecified_dims', unspecified_dims)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_sharding((input, sharding, unspecified_dims, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_sharding_eager_fallback(input, sharding=sharding, unspecified_dims=unspecified_dims, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_sharding, (), dict(input=input, sharding=sharding, unspecified_dims=unspecified_dims, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_sharding((input, sharding, unspecified_dims, name), None)
        if _result is not NotImplemented:
            return _result
    if sharding is None:
        sharding = ''
    sharding = _execute.make_str(sharding, 'sharding')
    if unspecified_dims is None:
        unspecified_dims = []
    if not isinstance(unspecified_dims, (list, tuple)):
        raise TypeError("Expected list for 'unspecified_dims' argument to 'xla_sharding' Op, not %r." % unspecified_dims)
    unspecified_dims = [_execute.make_int(_i, 'unspecified_dims') for _i in unspecified_dims]
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaSharding', input=input, sharding=sharding, unspecified_dims=unspecified_dims, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_sharding, (), dict(input=input, sharding=sharding, unspecified_dims=unspecified_dims, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'sharding', _op.get_attr('sharding'), 'unspecified_dims', _op.get_attr('unspecified_dims'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaSharding', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
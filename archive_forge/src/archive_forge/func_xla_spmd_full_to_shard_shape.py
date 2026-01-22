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
@tf_export('xla_spmd_full_to_shard_shape')
def xla_spmd_full_to_shard_shape(input: _atypes.TensorFuzzingAnnotation[TV_XlaSpmdFullToShardShape_T], manual_sharding: str, dim: int=-1, unspecified_dims=[], name=None) -> _atypes.TensorFuzzingAnnotation[TV_XlaSpmdFullToShardShape_T]:
    """An op used by XLA SPMD partitioner to switch from automatic partitioning to

  manual partitioning. It annotates the input (full-shape, to be automatically
  partitioned) with the same sharding used by manual partitioning, and outputs a
  shard-shaped tensor to be consumed by later manually-partitioned ops. If the
  shape is not evenly partitionable, the padding region will be masked with 0s.
  The conversion can happen partially in subgroups, by specifying the dim
  attribute, where only that dim will be converted.

  Args:
    input: A `Tensor`.
    manual_sharding: A `string`.
    dim: An optional `int`. Defaults to `-1`.
    unspecified_dims: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'XlaSpmdFullToShardShape', name, input, 'manual_sharding', manual_sharding, 'dim', dim, 'unspecified_dims', unspecified_dims)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_xla_spmd_full_to_shard_shape((input, manual_sharding, dim, unspecified_dims, name), None)
            if _result is not NotImplemented:
                return _result
            return xla_spmd_full_to_shard_shape_eager_fallback(input, manual_sharding=manual_sharding, dim=dim, unspecified_dims=unspecified_dims, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(xla_spmd_full_to_shard_shape, (), dict(input=input, manual_sharding=manual_sharding, dim=dim, unspecified_dims=unspecified_dims, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_xla_spmd_full_to_shard_shape((input, manual_sharding, dim, unspecified_dims, name), None)
        if _result is not NotImplemented:
            return _result
    manual_sharding = _execute.make_str(manual_sharding, 'manual_sharding')
    if dim is None:
        dim = -1
    dim = _execute.make_int(dim, 'dim')
    if unspecified_dims is None:
        unspecified_dims = []
    if not isinstance(unspecified_dims, (list, tuple)):
        raise TypeError("Expected list for 'unspecified_dims' argument to 'xla_spmd_full_to_shard_shape' Op, not %r." % unspecified_dims)
    unspecified_dims = [_execute.make_int(_i, 'unspecified_dims') for _i in unspecified_dims]
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('XlaSpmdFullToShardShape', input=input, manual_sharding=manual_sharding, dim=dim, unspecified_dims=unspecified_dims, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(xla_spmd_full_to_shard_shape, (), dict(input=input, manual_sharding=manual_sharding, dim=dim, unspecified_dims=unspecified_dims, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'manual_sharding', _op.get_attr('manual_sharding'), 'dim', _op._get_attr_int('dim'), 'unspecified_dims', _op.get_attr('unspecified_dims'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('XlaSpmdFullToShardShape', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
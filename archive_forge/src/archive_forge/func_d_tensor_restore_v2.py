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
@tf_export('d_tensor_restore_v2')
def d_tensor_restore_v2(prefix: _atypes.TensorFuzzingAnnotation[_atypes.String], tensor_names: _atypes.TensorFuzzingAnnotation[_atypes.String], shape_and_slices: _atypes.TensorFuzzingAnnotation[_atypes.String], input_shapes, input_layouts, dtypes, name=None):
    """TODO: add doc.

  Args:
    prefix: A `Tensor` of type `string`.
    tensor_names: A `Tensor` of type `string`.
    shape_and_slices: A `Tensor` of type `string`.
    input_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
    input_layouts: A list of `strings`.
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DTensorRestoreV2', name, prefix, tensor_names, shape_and_slices, 'input_shapes', input_shapes, 'input_layouts', input_layouts, 'dtypes', dtypes)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_d_tensor_restore_v2((prefix, tensor_names, shape_and_slices, input_shapes, input_layouts, dtypes, name), None)
            if _result is not NotImplemented:
                return _result
            return d_tensor_restore_v2_eager_fallback(prefix, tensor_names, shape_and_slices, input_shapes=input_shapes, input_layouts=input_layouts, dtypes=dtypes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(d_tensor_restore_v2, (), dict(prefix=prefix, tensor_names=tensor_names, shape_and_slices=shape_and_slices, input_shapes=input_shapes, input_layouts=input_layouts, dtypes=dtypes, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_d_tensor_restore_v2((prefix, tensor_names, shape_and_slices, input_shapes, input_layouts, dtypes, name), None)
        if _result is not NotImplemented:
            return _result
    if not isinstance(input_shapes, (list, tuple)):
        raise TypeError("Expected list for 'input_shapes' argument to 'd_tensor_restore_v2' Op, not %r." % input_shapes)
    input_shapes = [_execute.make_shape(_s, 'input_shapes') for _s in input_shapes]
    if not isinstance(input_layouts, (list, tuple)):
        raise TypeError("Expected list for 'input_layouts' argument to 'd_tensor_restore_v2' Op, not %r." % input_layouts)
    input_layouts = [_execute.make_str(_s, 'input_layouts') for _s in input_layouts]
    if not isinstance(dtypes, (list, tuple)):
        raise TypeError("Expected list for 'dtypes' argument to 'd_tensor_restore_v2' Op, not %r." % dtypes)
    dtypes = [_execute.make_type(_t, 'dtypes') for _t in dtypes]
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('DTensorRestoreV2', prefix=prefix, tensor_names=tensor_names, shape_and_slices=shape_and_slices, input_shapes=input_shapes, input_layouts=input_layouts, dtypes=dtypes, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(d_tensor_restore_v2, (), dict(prefix=prefix, tensor_names=tensor_names, shape_and_slices=shape_and_slices, input_shapes=input_shapes, input_layouts=input_layouts, dtypes=dtypes, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if not _result:
        return _op
    if _execute.must_record_gradient():
        _attrs = ('input_shapes', _op.get_attr('input_shapes'), 'input_layouts', _op.get_attr('input_layouts'), 'dtypes', _op.get_attr('dtypes'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DTensorRestoreV2', _inputs_flat, _attrs, _result)
    return _result
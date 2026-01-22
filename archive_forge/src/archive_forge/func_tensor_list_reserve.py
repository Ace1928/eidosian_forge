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
def tensor_list_reserve(element_shape: _atypes.TensorFuzzingAnnotation[TV_TensorListReserve_shape_type], num_elements: _atypes.TensorFuzzingAnnotation[_atypes.Int32], element_dtype: TV_TensorListReserve_element_dtype, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """List of the given size with empty elements.

  element_shape: the shape of the future elements of the list
  num_elements: the number of elements to reserve
  handle: the output list
  element_dtype: the desired type of elements in the list.

  Args:
    element_shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    num_elements: A `Tensor` of type `int32`.
    element_dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorListReserve', name, element_shape, num_elements, 'element_dtype', element_dtype)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_list_reserve_eager_fallback(element_shape, num_elements, element_dtype=element_dtype, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    element_dtype = _execute.make_type(element_dtype, 'element_dtype')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorListReserve', element_shape=element_shape, num_elements=num_elements, element_dtype=element_dtype, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('element_dtype', _op._get_attr_type('element_dtype'), 'shape_type', _op._get_attr_type('shape_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorListReserve', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
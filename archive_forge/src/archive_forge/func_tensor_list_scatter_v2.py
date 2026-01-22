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
def tensor_list_scatter_v2(tensor: _atypes.TensorFuzzingAnnotation[TV_TensorListScatterV2_element_dtype], indices: _atypes.TensorFuzzingAnnotation[_atypes.Int32], element_shape: _atypes.TensorFuzzingAnnotation[TV_TensorListScatterV2_shape_type], num_elements: _atypes.TensorFuzzingAnnotation[_atypes.Int32], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Creates a TensorList by indexing into a Tensor.

  Each member of the TensorList corresponds to one row of the input tensor,
  specified by the given index (see `tf.gather`).

  tensor: The input tensor.
  indices: The indices used to index into the list.
  element_shape: The shape of the elements in the list (can be less specified than
    the shape of the tensor).
  num_elements: The size of the output list. Must be large enough to accommodate
    the largest index in indices. If -1, the list is just large enough to include
    the largest index in indices.
  output_handle: The TensorList.

  Args:
    tensor: A `Tensor`.
    indices: A `Tensor` of type `int32`.
    element_shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    num_elements: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorListScatterV2', name, tensor, indices, element_shape, num_elements)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_list_scatter_v2_eager_fallback(tensor, indices, element_shape, num_elements, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorListScatterV2', tensor=tensor, indices=indices, element_shape=element_shape, num_elements=num_elements, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('element_dtype', _op._get_attr_type('element_dtype'), 'shape_type', _op._get_attr_type('shape_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorListScatterV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
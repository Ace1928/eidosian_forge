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
def tensor_list_stack(input_handle: _atypes.TensorFuzzingAnnotation[_atypes.Variant], element_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int32], element_dtype: TV_TensorListStack_element_dtype, num_elements: int=-1, name=None) -> _atypes.TensorFuzzingAnnotation[TV_TensorListStack_element_dtype]:
    """Stacks all tensors in the list.

  Requires that all tensors have the same shape.

  input_handle: the input list
  tensor: the gathered result
  num_elements: optional. If not -1, the number of elements in the list.

  Args:
    input_handle: A `Tensor` of type `variant`.
    element_shape: A `Tensor` of type `int32`.
    element_dtype: A `tf.DType`.
    num_elements: An optional `int`. Defaults to `-1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `element_dtype`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorListStack', name, input_handle, element_shape, 'element_dtype', element_dtype, 'num_elements', num_elements)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_list_stack_eager_fallback(input_handle, element_shape, element_dtype=element_dtype, num_elements=num_elements, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    element_dtype = _execute.make_type(element_dtype, 'element_dtype')
    if num_elements is None:
        num_elements = -1
    num_elements = _execute.make_int(num_elements, 'num_elements')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorListStack', input_handle=input_handle, element_shape=element_shape, element_dtype=element_dtype, num_elements=num_elements, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('element_dtype', _op._get_attr_type('element_dtype'), 'num_elements', _op._get_attr_int('num_elements'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorListStack', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
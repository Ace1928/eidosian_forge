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
def tensor_array_v3(size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], dtype: TV_TensorArrayV3_dtype, element_shape=None, dynamic_size: bool=False, clear_after_read: bool=True, identical_element_shapes: bool=False, tensor_array_name: str='', name=None):
    """An array of Tensors of given size.

  Write data via Write and read via Read or Pack.

  Args:
    size: A `Tensor` of type `int32`. The size of the array.
    dtype: A `tf.DType`. The type of the elements on the tensor_array.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `None`.
      The expected shape of an element, if known. Used to
      validate the shapes of TensorArray elements. If this shape is not
      fully specified, gathering zero-size TensorArrays is an error.
    dynamic_size: An optional `bool`. Defaults to `False`.
      A boolean that determines whether writes to the TensorArray
      are allowed to grow the size.  By default, this is not allowed.
    clear_after_read: An optional `bool`. Defaults to `True`.
      If true (default), Tensors in the TensorArray are cleared
      after being read.  This disables multiple read semantics but allows early
      release of memory.
    identical_element_shapes: An optional `bool`. Defaults to `False`.
      If true (default is false), then all
      elements in the TensorArray will be expected to have identical shapes.
      This allows certain behaviors, like dynamically checking for
      consistent shapes on write, and being able to fill in properly
      shaped zero tensors on stack -- even if the element_shape attribute
      is not fully defined.
    tensor_array_name: An optional `string`. Defaults to `""`.
      Overrides the name used for the temporary tensor_array
      resource. Default value is the name of the 'TensorArray' op (which
      is guaranteed unique).
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (handle, flow).

    handle: A `Tensor` of type `resource`.
    flow: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorArrayV3', name, size, 'dtype', dtype, 'element_shape', element_shape, 'dynamic_size', dynamic_size, 'clear_after_read', clear_after_read, 'identical_element_shapes', identical_element_shapes, 'tensor_array_name', tensor_array_name)
            _result = _TensorArrayV3Output._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_array_v3_eager_fallback(size, dtype=dtype, element_shape=element_shape, dynamic_size=dynamic_size, clear_after_read=clear_after_read, identical_element_shapes=identical_element_shapes, tensor_array_name=tensor_array_name, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    dtype = _execute.make_type(dtype, 'dtype')
    if element_shape is None:
        element_shape = None
    element_shape = _execute.make_shape(element_shape, 'element_shape')
    if dynamic_size is None:
        dynamic_size = False
    dynamic_size = _execute.make_bool(dynamic_size, 'dynamic_size')
    if clear_after_read is None:
        clear_after_read = True
    clear_after_read = _execute.make_bool(clear_after_read, 'clear_after_read')
    if identical_element_shapes is None:
        identical_element_shapes = False
    identical_element_shapes = _execute.make_bool(identical_element_shapes, 'identical_element_shapes')
    if tensor_array_name is None:
        tensor_array_name = ''
    tensor_array_name = _execute.make_str(tensor_array_name, 'tensor_array_name')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorArrayV3', size=size, dtype=dtype, element_shape=element_shape, dynamic_size=dynamic_size, clear_after_read=clear_after_read, identical_element_shapes=identical_element_shapes, tensor_array_name=tensor_array_name, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'element_shape', _op.get_attr('element_shape'), 'dynamic_size', _op._get_attr_bool('dynamic_size'), 'clear_after_read', _op._get_attr_bool('clear_after_read'), 'identical_element_shapes', _op._get_attr_bool('identical_element_shapes'), 'tensor_array_name', _op.get_attr('tensor_array_name'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorArrayV3', _inputs_flat, _attrs, _result)
    _result = _TensorArrayV3Output._make(_result)
    return _result
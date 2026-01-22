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
def stack_pop(handle: _atypes.TensorFuzzingAnnotation[_atypes.String], elem_type: TV_StackPop_elem_type, name=None) -> _atypes.TensorFuzzingAnnotation[TV_StackPop_elem_type]:
    """Deprecated, use StackPopV2.

  Args:
    handle: A `Tensor` of type mutable `string`.
    elem_type: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `elem_type`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("stack_pop op does not support eager execution. Arg 'handle' is a ref.")
    elem_type = _execute.make_type(elem_type, 'elem_type')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StackPop', handle=handle, elem_type=elem_type, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('elem_type', _op._get_attr_type('elem_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StackPop', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
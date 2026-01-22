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
def stack_pop_v2(handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], elem_type: TV_StackPopV2_elem_type, name=None) -> _atypes.TensorFuzzingAnnotation[TV_StackPopV2_elem_type]:
    """Pop the element at the top of the stack.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a stack.
    elem_type: A `tf.DType`. The type of the elem that is popped.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `elem_type`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StackPopV2', name, handle, 'elem_type', elem_type)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return stack_pop_v2_eager_fallback(handle, elem_type=elem_type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    elem_type = _execute.make_type(elem_type, 'elem_type')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StackPopV2', handle=handle, elem_type=elem_type, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('elem_type', _op._get_attr_type('elem_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StackPopV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
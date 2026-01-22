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
def stack_v2(max_size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], elem_type: TV_StackV2_elem_type, stack_name: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Resource]:
    """A stack that produces elements in first-in last-out order.

  Args:
    max_size: A `Tensor` of type `int32`.
      The maximum size of the stack if non-negative. If negative, the stack
      size is unlimited.
    elem_type: A `tf.DType`. The type of the elements on the stack.
    stack_name: An optional `string`. Defaults to `""`.
      Overrides the name used for the temporary stack resource. Default
      value is the name of the 'Stack' op (which is guaranteed unique).
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StackV2', name, max_size, 'elem_type', elem_type, 'stack_name', stack_name)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return stack_v2_eager_fallback(max_size, elem_type=elem_type, stack_name=stack_name, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    elem_type = _execute.make_type(elem_type, 'elem_type')
    if stack_name is None:
        stack_name = ''
    stack_name = _execute.make_str(stack_name, 'stack_name')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StackV2', max_size=max_size, elem_type=elem_type, stack_name=stack_name, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('elem_type', _op._get_attr_type('elem_type'), 'stack_name', _op.get_attr('stack_name'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StackV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
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
def ref_select(index: _atypes.TensorFuzzingAnnotation[_atypes.Int32], inputs: List[_atypes.TensorFuzzingAnnotation[TV_RefSelect_T]], name=None) -> _atypes.TensorFuzzingAnnotation[TV_RefSelect_T]:
    """Forwards the `index`th element of `inputs` to `output`.

  Args:
    index: A `Tensor` of type `int32`.
      A scalar that determines the input that gets selected.
    inputs: A list of at least 1 mutable `Tensor` objects with the same type.
      A list of ref tensors, one of which will be forwarded to `output`.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `inputs`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("ref_select op does not support eager execution. Arg 'output' is a ref.")
    if not isinstance(inputs, (list, tuple)):
        raise TypeError("Expected list for 'inputs' argument to 'ref_select' Op, not %r." % inputs)
    _attr_N = len(inputs)
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RefSelect', index=index, inputs=inputs, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'N', _op._get_attr_int('N'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RefSelect', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
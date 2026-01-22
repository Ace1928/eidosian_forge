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
def tensor_array_grad_v3(handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], flow_in: _atypes.TensorFuzzingAnnotation[_atypes.Float32], source: str, name=None):
    """Creates a TensorArray for storing the gradients of values in the given handle.

  If the given TensorArray gradient already exists, returns a reference to it.

  Locks the size of the original TensorArray by disabling its dynamic size flag.

  **A note about the input flow_in:**

  The handle flow_in forces the execution of the gradient lookup to occur
  only after certain other operations have occurred.  For example, when
  the forward TensorArray is dynamically sized, writes to this TensorArray
  may resize the object.  The gradient TensorArray is statically sized based
  on the size of the forward TensorArray when this operation executes.
  Furthermore, the size of the forward TensorArray is frozen by this call.
  As a result, the flow is used to ensure that the call to generate the gradient
  TensorArray only happens after all writes are executed.

  In the case of dynamically sized TensorArrays, gradient computation should
  only be performed on read operations that have themselves been chained via
  flow to occur only after all writes have executed. That way the final size
  of the forward TensorArray is known when this operation is called.

  **A note about the source attribute:**

  TensorArray gradient calls use an accumulator TensorArray object.  If
  multiple gradients are calculated and run in the same session, the multiple
  gradient nodes may accidentally flow through the same accumulator TensorArray.
  This double counts and generally breaks the TensorArray gradient flow.

  The solution is to identify which gradient call this particular
  TensorArray gradient is being called in.  This is performed by identifying
  a unique string (e.g. "gradients", "gradients_1", ...) from the input
  gradient Tensor's name.  This string is used as a suffix when creating
  the TensorArray gradient object here (the attribute `source`).

  The attribute `source` is added as a suffix to the forward TensorArray's
  name when performing the creation / lookup, so that each separate gradient
  calculation gets its own TensorArray accumulator.

  Args:
    handle: A `Tensor` of type `resource`.
      The handle to the forward TensorArray.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    source: A `string`.
      The gradient source string, used to decide which gradient TensorArray
      to return.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (grad_handle, flow_out).

    grad_handle: A `Tensor` of type `resource`.
    flow_out: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorArrayGradV3', name, handle, flow_in, 'source', source)
            _result = _TensorArrayGradV3Output._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_array_grad_v3_eager_fallback(handle, flow_in, source=source, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    source = _execute.make_str(source, 'source')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorArrayGradV3', handle=handle, flow_in=flow_in, source=source, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('source', _op.get_attr('source'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorArrayGradV3', _inputs_flat, _attrs, _result)
    _result = _TensorArrayGradV3Output._make(_result)
    return _result
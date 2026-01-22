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
def tpu_replicated_input(inputs: List[_atypes.TensorFuzzingAnnotation[TV_TPUReplicatedInput_T]], is_mirrored_variable: bool=False, index: int=-1, is_packed: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_TPUReplicatedInput_T]:
    """Connects N inputs to an N-way replicated TPU computation.

  This operation holds a replicated input to a `tpu.replicate()` computation subgraph.
  Each replicated input has the same shape and type alongside the output.

  For example:
  ```
  %a = "tf.opA"()
  %b = "tf.opB"()
  %replicated_input = "tf.TPUReplicatedInput"(%a, %b)
  %computation = "tf.Computation"(%replicated_input)
  ```
  The above computation has a replicated input of two replicas.

  Args:
    inputs: A list of at least 1 `Tensor` objects with the same type.
    is_mirrored_variable: An optional `bool`. Defaults to `False`.
    index: An optional `int`. Defaults to `-1`.
    is_packed: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `inputs`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TPUReplicatedInput', name, inputs, 'is_mirrored_variable', is_mirrored_variable, 'index', index, 'is_packed', is_packed)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tpu_replicated_input_eager_fallback(inputs, is_mirrored_variable=is_mirrored_variable, index=index, is_packed=is_packed, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(inputs, (list, tuple)):
        raise TypeError("Expected list for 'inputs' argument to 'tpu_replicated_input' Op, not %r." % inputs)
    _attr_N = len(inputs)
    if is_mirrored_variable is None:
        is_mirrored_variable = False
    is_mirrored_variable = _execute.make_bool(is_mirrored_variable, 'is_mirrored_variable')
    if index is None:
        index = -1
    index = _execute.make_int(index, 'index')
    if is_packed is None:
        is_packed = False
    is_packed = _execute.make_bool(is_packed, 'is_packed')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TPUReplicatedInput', inputs=inputs, is_mirrored_variable=is_mirrored_variable, index=index, is_packed=is_packed, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('N', _op._get_attr_int('N'), 'T', _op._get_attr_type('T'), 'is_mirrored_variable', _op._get_attr_bool('is_mirrored_variable'), 'index', _op._get_attr_int('index'), 'is_packed', _op._get_attr_bool('is_packed'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TPUReplicatedInput', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
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
def case(branch_index: _atypes.TensorFuzzingAnnotation[_atypes.Int32], input, Tout, branches, output_shapes=[], name=None):
    """An n-way switch statement which calls a single branch function.

      An n-way switch statement, implementing the following:
      ```
      switch (branch_index) {
        case 0:
          output = branches[0](input);
          break;
        case 1:
          output = branches[1](input);
          break;
        ...
        case [[nbranches-1]]:
        default:
          output = branches[nbranches-1](input);
          break;
      }
      ```

  Args:
    branch_index: A `Tensor` of type `int32`.
      The branch selector, an int32 Tensor.
    input: A list of `Tensor` objects.
      A list of input tensors passed to the branch function.
    Tout: A list of `tf.DTypes`. A list of output types.
    branches: A list of functions decorated with @Defun that has length `>= 1`.
            A list of functions each of which takes 'inputs' and returns a list of
            tensors, whose types are the same as what every other branch returns.
    output_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Case', name, branch_index, input, 'Tout', Tout, 'branches', branches, 'output_shapes', output_shapes)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return case_eager_fallback(branch_index, input, Tout=Tout, branches=branches, output_shapes=output_shapes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(Tout, (list, tuple)):
        raise TypeError("Expected list for 'Tout' argument to 'case' Op, not %r." % Tout)
    Tout = [_execute.make_type(_t, 'Tout') for _t in Tout]
    if not isinstance(branches, (list, tuple)):
        raise TypeError("Expected list for 'branches' argument to 'case' Op, not %r." % branches)
    if output_shapes is None:
        output_shapes = []
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'case' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('Case', branch_index=branch_index, input=input, Tout=Tout, branches=branches, output_shapes=output_shapes, name=name)
    _result = _outputs[:]
    if not _result:
        return _op
    if _execute.must_record_gradient():
        _attrs = ('Tin', _op.get_attr('Tin'), 'Tout', _op.get_attr('Tout'), 'branches', _op.get_attr('branches'), 'output_shapes', _op.get_attr('output_shapes'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('Case', _inputs_flat, _attrs, _result)
    return _result
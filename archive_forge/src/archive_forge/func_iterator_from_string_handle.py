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
def iterator_from_string_handle(string_handle: _atypes.TensorFuzzingAnnotation[_atypes.String], output_types=[], output_shapes=[], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Resource]:
    """Converts the given string representing a handle to an iterator to a resource.

  Args:
    string_handle: A `Tensor` of type `string`.
      A string representation of the given handle.
    output_types: An optional list of `tf.DTypes`. Defaults to `[]`.
      If specified, defines the type of each tuple component in an
      element produced by the resulting iterator.
    output_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      If specified, defines the shape of each tuple component in an
      element produced by the resulting iterator.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'IteratorFromStringHandle', name, string_handle, 'output_types', output_types, 'output_shapes', output_shapes)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return iterator_from_string_handle_eager_fallback(string_handle, output_types=output_types, output_shapes=output_shapes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if output_types is None:
        output_types = []
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'iterator_from_string_handle' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if output_shapes is None:
        output_shapes = []
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'iterator_from_string_handle' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('IteratorFromStringHandle', string_handle=string_handle, output_types=output_types, output_shapes=output_shapes, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('IteratorFromStringHandle', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
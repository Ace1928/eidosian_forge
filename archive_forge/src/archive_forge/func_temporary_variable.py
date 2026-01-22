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
def temporary_variable(shape, dtype: TV_TemporaryVariable_dtype, var_name: str='', name=None) -> _atypes.TensorFuzzingAnnotation[TV_TemporaryVariable_dtype]:
    """Returns a tensor that may be mutated, but only persists within a single step.

  This is an experimental op for internal use only and it is possible to use this
  op in unsafe ways.  DO NOT USE unless you fully understand the risks.

  It is the caller's responsibility to ensure that 'ref' is eventually passed to a
  matching 'DestroyTemporaryVariable' op after all other uses have completed.

  Outputs a ref to the tensor state so it may be read or modified.

    E.g.
        var = state_ops._temporary_variable([1, 2], types.float_)
        var_name = var.op.name
        var = state_ops.assign(var, [[4.0, 5.0]])
        var = state_ops.assign_add(var, [[6.0, 7.0]])
        final = state_ops._destroy_temporary_variable(var, var_name=var_name)

  Args:
    shape: A `tf.TensorShape` or list of `ints`.
      The shape of the variable tensor.
    dtype: A `tf.DType`. The type of elements in the variable tensor.
    var_name: An optional `string`. Defaults to `""`.
      Overrides the name used for the temporary variable resource. Default
      value is the name of the 'TemporaryVariable' op (which is guaranteed unique).
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor` of type `dtype`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("temporary_variable op does not support eager execution. Arg 'ref' is a ref.")
    shape = _execute.make_shape(shape, 'shape')
    dtype = _execute.make_type(dtype, 'dtype')
    if var_name is None:
        var_name = ''
    var_name = _execute.make_str(var_name, 'var_name')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TemporaryVariable', shape=shape, dtype=dtype, var_name=var_name, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('shape', _op.get_attr('shape'), 'dtype', _op._get_attr_type('dtype'), 'var_name', _op.get_attr('var_name'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TemporaryVariable', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
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
def ref_enter(data: _atypes.TensorFuzzingAnnotation[TV_RefEnter_T], frame_name: str, is_constant: bool=False, parallel_iterations: int=10, name=None) -> _atypes.TensorFuzzingAnnotation[TV_RefEnter_T]:
    """Creates or finds a child frame, and makes `data` available to the child frame.

  The unique `frame_name` is used by the `Executor` to identify frames. If
  `is_constant` is true, `output` is a constant in the child frame; otherwise
  it may be changed in the child frame. At most `parallel_iterations` iterations
  are run in parallel in the child frame.

  Args:
    data: A mutable `Tensor`.
      The tensor to be made available to the child frame.
    frame_name: A `string`. The name of the child frame.
    is_constant: An optional `bool`. Defaults to `False`.
      If true, the output is constant within the child frame.
    parallel_iterations: An optional `int`. Defaults to `10`.
      The number of iterations allowed to run in parallel.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `data`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("ref_enter op does not support eager execution. Arg 'output' is a ref.")
    frame_name = _execute.make_str(frame_name, 'frame_name')
    if is_constant is None:
        is_constant = False
    is_constant = _execute.make_bool(is_constant, 'is_constant')
    if parallel_iterations is None:
        parallel_iterations = 10
    parallel_iterations = _execute.make_int(parallel_iterations, 'parallel_iterations')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RefEnter', data=data, frame_name=frame_name, is_constant=is_constant, parallel_iterations=parallel_iterations, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'frame_name', _op.get_attr('frame_name'), 'is_constant', _op._get_attr_bool('is_constant'), 'parallel_iterations', _op._get_attr_int('parallel_iterations'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RefEnter', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
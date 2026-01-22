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
def parallel_filter_dataset(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], other_arguments, num_parallel_calls: _atypes.TensorFuzzingAnnotation[_atypes.Int64], predicate, output_types, output_shapes, deterministic: str='default', metadata: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Creates a dataset containing elements of `input_dataset` matching `predicate`.

  The `predicate` function must return a scalar boolean and accept the
  following arguments:

  * One tensor for each component of an element of `input_dataset`.
  * One tensor for each value in `other_arguments`.

  Unlike a "FilterDataset", which applies `predicate` sequentially, this dataset
  invokes up to `num_parallel_calls` copies of `predicate` in parallel.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when
      building a closure for `predicate`.
    num_parallel_calls: A `Tensor` of type `int64`.
      The number of concurrent invocations of `predicate` that process
      elements from `input_dataset` in parallel.
    predicate: A function decorated with @Defun.
      A function returning a scalar boolean.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    deterministic: An optional `string`. Defaults to `"default"`.
      A string indicating the op-level determinism to use. Deterministic controls
      whether the interleave is allowed to return elements out of order if the next
      element to be returned isn't available, but a later element is. Options are
      "true", "false", and "default". "default" indicates that determinism should be
      decided by the `experimental_deterministic` parameter of `tf.data.Options`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ParallelFilterDataset', name, input_dataset, other_arguments, num_parallel_calls, 'predicate', predicate, 'deterministic', deterministic, 'output_types', output_types, 'output_shapes', output_shapes, 'metadata', metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return parallel_filter_dataset_eager_fallback(input_dataset, other_arguments, num_parallel_calls, predicate=predicate, deterministic=deterministic, output_types=output_types, output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'parallel_filter_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'parallel_filter_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if deterministic is None:
        deterministic = 'default'
    deterministic = _execute.make_str(deterministic, 'deterministic')
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ParallelFilterDataset', input_dataset=input_dataset, other_arguments=other_arguments, num_parallel_calls=num_parallel_calls, predicate=predicate, output_types=output_types, output_shapes=output_shapes, deterministic=deterministic, metadata=metadata, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('predicate', _op.get_attr('predicate'), 'deterministic', _op.get_attr('deterministic'), 'Targuments', _op.get_attr('Targuments'), 'output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'), 'metadata', _op.get_attr('metadata'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ParallelFilterDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
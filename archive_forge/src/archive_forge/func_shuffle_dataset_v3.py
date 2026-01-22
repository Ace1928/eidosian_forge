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
def shuffle_dataset_v3(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], buffer_size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], seed: _atypes.TensorFuzzingAnnotation[_atypes.Int64], seed2: _atypes.TensorFuzzingAnnotation[_atypes.Int64], seed_generator: _atypes.TensorFuzzingAnnotation[_atypes.Resource], output_types, output_shapes, reshuffle_each_iteration: bool=True, metadata: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    buffer_size: A `Tensor` of type `int64`.
    seed: A `Tensor` of type `int64`.
    seed2: A `Tensor` of type `int64`.
    seed_generator: A `Tensor` of type `resource`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    reshuffle_each_iteration: An optional `bool`. Defaults to `True`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ShuffleDatasetV3', name, input_dataset, buffer_size, seed, seed2, seed_generator, 'reshuffle_each_iteration', reshuffle_each_iteration, 'output_types', output_types, 'output_shapes', output_shapes, 'metadata', metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return shuffle_dataset_v3_eager_fallback(input_dataset, buffer_size, seed, seed2, seed_generator, reshuffle_each_iteration=reshuffle_each_iteration, output_types=output_types, output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'shuffle_dataset_v3' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'shuffle_dataset_v3' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if reshuffle_each_iteration is None:
        reshuffle_each_iteration = True
    reshuffle_each_iteration = _execute.make_bool(reshuffle_each_iteration, 'reshuffle_each_iteration')
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ShuffleDatasetV3', input_dataset=input_dataset, buffer_size=buffer_size, seed=seed, seed2=seed2, seed_generator=seed_generator, output_types=output_types, output_shapes=output_shapes, reshuffle_each_iteration=reshuffle_each_iteration, metadata=metadata, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('reshuffle_each_iteration', _op._get_attr_bool('reshuffle_each_iteration'), 'output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'), 'metadata', _op.get_attr('metadata'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ShuffleDatasetV3', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
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
def parse_example_dataset_v2(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], num_parallel_calls: _atypes.TensorFuzzingAnnotation[_atypes.Int64], dense_defaults, sparse_keys, dense_keys, sparse_types, dense_shapes, output_types, output_shapes, deterministic: str='default', ragged_keys=[], ragged_value_types=[], ragged_split_types=[], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Transforms `input_dataset` containing `Example` protos as vectors of DT_STRING into a dataset of `Tensor` or `SparseTensor` objects representing the parsed features.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    num_parallel_calls: A `Tensor` of type `int64`.
    dense_defaults: A list of `Tensor` objects with types from: `float32`, `int64`, `string`.
      A dict mapping string keys to `Tensor`s.
      The keys of the dict must match the dense_keys of the feature.
    sparse_keys: A list of `strings`.
      A list of string keys in the examples features.
      The results for these keys will be returned as `SparseTensor` objects.
    dense_keys: A list of `strings`.
      A list of Ndense string Tensors (scalars).
      The keys expected in the Examples features associated with dense values.
    sparse_types: A list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`.
      A list of `DTypes` of the same length as `sparse_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
      and `tf.string` (`BytesList`) are supported.
    dense_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      List of tuples with the same length as `dense_keys`.
      The shape of the data for each dense feature referenced by `dense_keys`.
      Required for any input tensors identified by `dense_keys`.  Must be
      either fully defined, or may contain an unknown first dimension.
      An unknown first dimension means the feature is treated as having
      a variable number of blocks, and the output shape along this dimension
      is considered unknown at graph build time.  Padding is applied for
      minibatch elements smaller than the maximum number of blocks for the
      given feature along this dimension.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
      The type list for the return values.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
      The list of shapes being produced.
    deterministic: An optional `string`. Defaults to `"default"`.
      A string indicating the op-level determinism to use. Deterministic controls
      whether the dataset is allowed to return elements out of order if the next
      element to be returned isn't available, but a later element is. Options are
      "true", "false", and "default". "default" indicates that determinism should be
      decided by the `experimental_deterministic` parameter of `tf.data.Options`.
    ragged_keys: An optional list of `strings`. Defaults to `[]`.
    ragged_value_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
    ragged_split_types: An optional list of `tf.DTypes` from: `tf.int32, tf.int64`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ParseExampleDatasetV2', name, input_dataset, num_parallel_calls, dense_defaults, 'sparse_keys', sparse_keys, 'dense_keys', dense_keys, 'sparse_types', sparse_types, 'dense_shapes', dense_shapes, 'output_types', output_types, 'output_shapes', output_shapes, 'deterministic', deterministic, 'ragged_keys', ragged_keys, 'ragged_value_types', ragged_value_types, 'ragged_split_types', ragged_split_types)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return parse_example_dataset_v2_eager_fallback(input_dataset, num_parallel_calls, dense_defaults, sparse_keys=sparse_keys, dense_keys=dense_keys, sparse_types=sparse_types, dense_shapes=dense_shapes, output_types=output_types, output_shapes=output_shapes, deterministic=deterministic, ragged_keys=ragged_keys, ragged_value_types=ragged_value_types, ragged_split_types=ragged_split_types, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(sparse_keys, (list, tuple)):
        raise TypeError("Expected list for 'sparse_keys' argument to 'parse_example_dataset_v2' Op, not %r." % sparse_keys)
    sparse_keys = [_execute.make_str(_s, 'sparse_keys') for _s in sparse_keys]
    if not isinstance(dense_keys, (list, tuple)):
        raise TypeError("Expected list for 'dense_keys' argument to 'parse_example_dataset_v2' Op, not %r." % dense_keys)
    dense_keys = [_execute.make_str(_s, 'dense_keys') for _s in dense_keys]
    if not isinstance(sparse_types, (list, tuple)):
        raise TypeError("Expected list for 'sparse_types' argument to 'parse_example_dataset_v2' Op, not %r." % sparse_types)
    sparse_types = [_execute.make_type(_t, 'sparse_types') for _t in sparse_types]
    if not isinstance(dense_shapes, (list, tuple)):
        raise TypeError("Expected list for 'dense_shapes' argument to 'parse_example_dataset_v2' Op, not %r." % dense_shapes)
    dense_shapes = [_execute.make_shape(_s, 'dense_shapes') for _s in dense_shapes]
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'parse_example_dataset_v2' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'parse_example_dataset_v2' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if deterministic is None:
        deterministic = 'default'
    deterministic = _execute.make_str(deterministic, 'deterministic')
    if ragged_keys is None:
        ragged_keys = []
    if not isinstance(ragged_keys, (list, tuple)):
        raise TypeError("Expected list for 'ragged_keys' argument to 'parse_example_dataset_v2' Op, not %r." % ragged_keys)
    ragged_keys = [_execute.make_str(_s, 'ragged_keys') for _s in ragged_keys]
    if ragged_value_types is None:
        ragged_value_types = []
    if not isinstance(ragged_value_types, (list, tuple)):
        raise TypeError("Expected list for 'ragged_value_types' argument to 'parse_example_dataset_v2' Op, not %r." % ragged_value_types)
    ragged_value_types = [_execute.make_type(_t, 'ragged_value_types') for _t in ragged_value_types]
    if ragged_split_types is None:
        ragged_split_types = []
    if not isinstance(ragged_split_types, (list, tuple)):
        raise TypeError("Expected list for 'ragged_split_types' argument to 'parse_example_dataset_v2' Op, not %r." % ragged_split_types)
    ragged_split_types = [_execute.make_type(_t, 'ragged_split_types') for _t in ragged_split_types]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ParseExampleDatasetV2', input_dataset=input_dataset, num_parallel_calls=num_parallel_calls, dense_defaults=dense_defaults, sparse_keys=sparse_keys, dense_keys=dense_keys, sparse_types=sparse_types, dense_shapes=dense_shapes, output_types=output_types, output_shapes=output_shapes, deterministic=deterministic, ragged_keys=ragged_keys, ragged_value_types=ragged_value_types, ragged_split_types=ragged_split_types, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('sparse_keys', _op.get_attr('sparse_keys'), 'dense_keys', _op.get_attr('dense_keys'), 'sparse_types', _op.get_attr('sparse_types'), 'Tdense', _op.get_attr('Tdense'), 'dense_shapes', _op.get_attr('dense_shapes'), 'output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'), 'deterministic', _op.get_attr('deterministic'), 'ragged_keys', _op.get_attr('ragged_keys'), 'ragged_value_types', _op.get_attr('ragged_value_types'), 'ragged_split_types', _op.get_attr('ragged_split_types'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ParseExampleDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result
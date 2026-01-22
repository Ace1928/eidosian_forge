from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('data.experimental.parse_example_dataset')
@deprecation.deprecated(None, 'Use `tf.data.Dataset.map(tf.io.parse_example(...))` instead.')
def parse_example_dataset(features, num_parallel_calls=1, deterministic=None):
    """A transformation that parses `Example` protos into a `dict` of tensors.

  Parses a number of serialized `Example` protos given in `serialized`. We refer
  to `serialized` as a batch with `batch_size` many entries of individual
  `Example` protos.

  This op parses serialized examples into a dictionary mapping keys to `Tensor`,
  `SparseTensor`, and `RaggedTensor` objects. `features` is a dict from keys to
  `VarLenFeature`, `RaggedFeature`, `SparseFeature`, and `FixedLenFeature`
  objects. Each `VarLenFeature` and `SparseFeature` is mapped to a
  `SparseTensor`; each `RaggedFeature` is mapped to a `RaggedTensor`; and each
  `FixedLenFeature` is mapped to a `Tensor`. See `tf.io.parse_example` for more
  details about feature dictionaries.

  Args:
   features: A `dict` mapping feature keys to `FixedLenFeature`,
     `VarLenFeature`, `RaggedFeature`, and `SparseFeature` values.
   num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
      representing the number of parsing processes to call in parallel.
   deterministic: (Optional.) A boolean controlling whether determinism
      should be traded for performance by allowing elements to be produced out
      of order if some parsing calls complete faster than others. If
      `deterministic` is `None`, the
      `tf.data.Options.deterministic` dataset option (`True` by default) is used
      to decide whether to produce elements deterministically.

  Returns:
    A dataset transformation function, which can be passed to
    `tf.data.Dataset.apply`.

  Raises:
    ValueError: if features argument is None.
  """
    if features is None:
        raise ValueError('Argument `features` is required, but not specified.')

    def _apply_fn(dataset):
        """Function from `Dataset` to `Dataset` that applies the transformation."""
        out_dataset = _ParseExampleDataset(dataset, features, num_parallel_calls, deterministic)
        if any((isinstance(feature, parsing_ops.SparseFeature) or isinstance(feature, parsing_ops.RaggedFeature) for feature in features.values())):
            out_dataset = out_dataset.map(lambda x: parsing_ops._construct_tensors_for_composite_features(features, x), num_parallel_calls=num_parallel_calls)
        return out_dataset
    return _apply_fn
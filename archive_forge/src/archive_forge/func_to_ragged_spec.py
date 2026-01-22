from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.ops.ragged import ragged_tensor
def to_ragged_spec(spec):
    """Returns the new spec based on RaggedTensors."""
    if not isinstance(spec, tensor.TensorSpec) or spec.shape.rank is None or spec.shape.is_fully_defined():
        return spec
    else:
        ragged_rank = max([axis for axis, size in enumerate(spec.shape.as_list()) if size is None])
        return ragged_tensor.RaggedTensorSpec(shape=spec.shape, dtype=spec.dtype, ragged_rank=ragged_rank, row_splits_dtype=row_splits_dtype)
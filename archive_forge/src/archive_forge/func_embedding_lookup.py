from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
@dispatch.dispatch_for_api(embedding_ops.embedding_lookup)
def embedding_lookup(params, ids: ragged_tensor.Ragged, partition_strategy='mod', name=None, validate_indices=True, max_norm=None):
    """Look up the ragged ids in a list of embedding tensors.

  Args:
    params: A tensor representing the complete embedding tensor having the shape
      [e1, ...eM]
    ragged_ids: A 'RaggedTensor' with type 'int32' or 'int64' containing the ids
      to be looked up in 'params' of shape [r0, ..rN]. Values must be in the
      range '[0, params.shape[0]]'.
    partition_strategy: A string specifying the partitioning strategy.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value.
    name: A name for the operation (optional)

  Returns:
    A ragged tensor of shape [r0, r1, ...rN, e1, ...eM].

  Raises:
    ValueError: When params is empty or the type of the ids is not int32 or
      int64.
  """
    if params is None:
        raise ValueError('params must be specified.')
    if isinstance(params, (list, tuple)) and (not params):
        raise ValueError('params should not be empty.')
    if ids.dtype != dtypes.int32 and ids.dtype != dtypes.int64:
        raise ValueError(f'The values contained by the inputs have type {str(ids.dtype)} and cannot be processed. All values should be indices, either of type `int32` or `int64`.')
    with ops.name_scope(name, 'embedding_lookup_ragged') as name:
        looked_up_ragged = ragged_functional_ops.map_flat_values(embedding_ops.embedding_lookup, params=params, ids=ids, partition_strategy=partition_strategy, max_norm=max_norm)
        return looked_up_ragged
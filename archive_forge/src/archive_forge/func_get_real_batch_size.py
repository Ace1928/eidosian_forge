import numpy as np
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
def get_real_batch_size(self, dataset_batch):
    """Returns the number of elements in a potentially partial batch."""
    if isinstance(dataset_batch, (tuple, list)):
        dataset_batch = dataset_batch[0]
    assert nest.flatten(dataset_batch)

    def _find_any_tensor(batch_features):
        tensors = [x for x in nest.flatten(batch_features) if tensor_util.is_tf_type(x)]
        if not tensors:
            raise ValueError('Cannot find any Tensor in features dict.')
        return tensors[0]
    return backend.cast(backend.shape(_find_any_tensor(dataset_batch))[0], dtype='int64')
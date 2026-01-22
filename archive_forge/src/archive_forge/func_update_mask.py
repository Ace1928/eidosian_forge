import numpy as np
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
def update_mask(self, padding_mask, dataset_batch):
    """Calculate and cache the amount of padding required for a batch."""
    original_batch_size = self.get_real_batch_size(dataset_batch)
    missing_count = self.padded_batch_size - original_batch_size
    mask = backend.concatenate([array_ops.ones(original_batch_size), array_ops.zeros(missing_count)], axis=0)
    return backend.concatenate([padding_mask, mask], axis=0)
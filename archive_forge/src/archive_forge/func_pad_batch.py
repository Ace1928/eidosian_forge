import numpy as np
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
def pad_batch(self, *dataset_batch_elements):
    """Pads out the batch dimension of a tensor to the complete batch size."""

    def _pad(batch):
        """Helper function to pad nested data within each batch elements."""
        padded_dict_batch = {}
        if isinstance(batch, dict):
            for key, value in batch.items():
                padded_dict_batch[key] = _pad(value)
            return padded_dict_batch
        rank = len(batch.shape)
        assert rank > 0
        missing_count = self.padded_batch_size - self.get_real_batch_size(batch)
        padding = backend.stack([[0, missing_count]] + [[0, 0]] * (rank - 1))
        return array_ops.pad(batch, padding, 'constant')
    if len(dataset_batch_elements) == 1:
        return _pad(dataset_batch_elements[0])
    batch_elements = []
    for batch_element in dataset_batch_elements:
        batch_elements.append(_pad(batch_element))
    return tuple(batch_elements)
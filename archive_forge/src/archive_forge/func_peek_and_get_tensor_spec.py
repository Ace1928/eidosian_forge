import numpy as np
import tree
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
def peek_and_get_tensor_spec(self):
    from keras.src.utils.module_utils import tensorflow as tf
    batch_data = next(iter(self._dataloader))

    def get_tensor_spec(x):
        shape = x.shape
        if len(shape) < 1:
            raise ValueError(f'When passing a Pytorch DataLoader to a Keras model, the arrays returned by the generator must be at least rank 1. Received: {x} of rank {len(x.shape)}')
        shape = list(shape)
        shape[0] = None
        dtype = backend.standardize_dtype(x.dtype)
        return tf.TensorSpec(shape=shape, dtype=dtype)
    return tuple(tree.map_structure(get_tensor_spec, batch_data))
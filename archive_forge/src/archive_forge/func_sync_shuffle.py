import math
import numpy as np
import tree
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.utils.dataset_utils import is_torch_tensor
def sync_shuffle(data, num_samples=None):
    if num_samples is None:
        num_samples_set = set((int(i.shape[0]) for i in tree.flatten(data)))
        assert len(num_samples_set) == 1
        num_samples = num_samples_set.pop()
    p = np.random.permutation(num_samples)
    return tree.map_structure(lambda x: x[p], data)
import math
import numpy as np
import tree
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.utils.dataset_utils import is_torch_tensor
def class_weight_to_sample_weights(y, class_weight):
    sample_weight = np.ones(shape=(y.shape[0],), dtype=backend.floatx())
    if len(y.shape) > 1:
        if y.shape[-1] != 1:
            y = np.argmax(y, axis=-1)
        else:
            y = np.squeeze(y, axis=-1)
    y = np.round(y).astype('int32')
    for i in range(y.shape[0]):
        sample_weight[i] = class_weight.get(int(y[i]), 1.0)
    return sample_weight
import numpy as np
import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.utils.nest import pack_sequence_as
def convert_keras_tensor_to_numpy(x, fill_value=None):
    if isinstance(x, KerasTensor):
        shape = list(x.shape)
        if fill_value:
            for i, e in enumerate(shape):
                if e is None:
                    shape[i] = fill_value
        return np.empty(shape=shape, dtype=x.dtype)
    return x
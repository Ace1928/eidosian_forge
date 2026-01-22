import collections
import math
import numpy as np
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils import tree
def can_slice_array(x):
    return x is None or isinstance(x, ARRAY_TYPES) or data_adapter_utils.is_tensorflow_tensor(x) or data_adapter_utils.is_jax_array(x) or data_adapter_utils.is_torch_tensor(x) or data_adapter_utils.is_scipy_sparse(x) or hasattr(x, '__array__')
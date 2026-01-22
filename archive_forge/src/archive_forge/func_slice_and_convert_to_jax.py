import functools
import math
import numpy as np
from keras.src.trainers.data_adapters import array_slicing
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
from keras.src.utils import tree
def slice_and_convert_to_jax(sliceable, indices=None):
    x = sliceable[indices]
    x = sliceable.convert_to_jax_compatible(x)
    x = convert_to_tensor(x)
    return x
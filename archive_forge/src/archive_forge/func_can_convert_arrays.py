import math
import numpy as np
import tree
from keras.src import backend
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
from keras.src.utils.dataset_utils import is_torch_tensor
from keras.src.utils.nest import lists_to_tuples
def can_convert_arrays(arrays):
    """Check if array like-inputs can be handled by `ArrayDataAdapter`

    Args:
        inputs: Structure of `Tensor`s, NumPy arrays, or tensor-like.

    Returns:
        `True` if `arrays` can be handled by `ArrayDataAdapter`, `False`
        otherwise.
    """

    def can_convert_single_array(x):
        is_none = x is None
        known_type = isinstance(x, data_adapter_utils.ARRAY_TYPES)
        convertable_type = hasattr(x, '__array__')
        return is_none or known_type or convertable_type
    return all(tree.flatten(tree.map_structure(can_convert_single_array, arrays)))
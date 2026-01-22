import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
@staticmethod
@check_framework_is_available
def random_int_tensor(shape: List[int], max_value: int, min_value: int=0, framework: str='pt', dtype: str='int64'):
    """
        Generates a tensor of random integers in the [min_value, max_value) range.

        Args:
            shape (`List[int]`):
                The shape of the random tensor.
            max_value (`int`):
                The maximum value allowed.
            min_value (`int`, defaults to 0):
                The minimum value allowed.
            framework (`str`, defaults to `"pt"`):
                The requested framework.
            dtype (`str`, defaults to `"int64"`):
                The dtype of the generated integer tensor. Could be "int64", "int32", "int8".

        Returns:
            A random tensor in the requested framework.
        """
    if framework == 'pt':
        return torch.randint(low=min_value, high=max_value, size=shape, dtype=DTYPE_MAPPER.pt(dtype))
    elif framework == 'tf':
        return tf.random.uniform(shape, minval=min_value, maxval=max_value, dtype=DTYPE_MAPPER.tf(dtype))
    else:
        return np.random.randint(min_value, high=max_value, size=shape, dtype=DTYPE_MAPPER.np(dtype))
import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
@staticmethod
@check_framework_is_available
def random_float_tensor(shape: List[int], min_value: float=0, max_value: float=1, framework: str='pt', dtype: str='fp32'):
    """
        Generates a tensor of random floats in the [min_value, max_value) range.

        Args:
            shape (`List[int]`):
                The shape of the random tensor.
            min_value (`float`, defaults to 0):
                The minimum value allowed.
            max_value (`float`, defaults to 1):
                The maximum value allowed.
            framework (`str`, defaults to `"pt"`):
                The requested framework.
            dtype (`str`, defaults to `"fp32"`):
                The dtype of the generated float tensor. Could be "fp32", "fp16", "bf16".

        Returns:
            A random tensor in the requested framework.
        """
    if framework == 'pt':
        tensor = torch.empty(shape, dtype=DTYPE_MAPPER.pt(dtype)).uniform_(min_value, max_value)
        return tensor
    elif framework == 'tf':
        return tf.random.uniform(shape, minval=min_value, maxval=max_value, dtype=DTYPE_MAPPER.tf(dtype))
    else:
        return np.random.uniform(low=min_value, high=max_value, size=shape).astype(DTYPE_MAPPER.np(dtype))
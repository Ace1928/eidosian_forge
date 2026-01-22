import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
@classmethod
def concat_inputs(cls, inputs, dim: int):
    """
        Concatenates inputs together.

        Args:
            inputs:
                The list of tensors in a given framework to concatenate.
            dim (`int`):
                The dimension along which to concatenate.
        Returns:
            The tensor of the concatenation.
        """
    if not inputs:
        raise ValueError('You did not provide any inputs to concat')
    framework = cls._infer_framework_from_input(inputs[0])
    if framework == 'pt':
        return torch.cat(inputs, dim=dim)
    elif framework == 'tf':
        return tf.concat(inputs, axis=dim)
    else:
        return np.concatenate(inputs, axis=dim)
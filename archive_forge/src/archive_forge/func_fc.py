from collections import OrderedDict
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
import tree  # pip install dm_tree
from types import MappingProxyType
from typing import List, Optional
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import SpaceStruct, TensorType, TensorStructType, Union
@PublicAPI
def fc(x: np.ndarray, weights: np.ndarray, biases: Optional[np.ndarray]=None, framework: Optional[str]=None) -> np.ndarray:
    """Calculates FC (dense) layer outputs given weights/biases and input.

    Args:
        x: The input to the dense layer.
        weights: The weights matrix.
        biases: The biases vector. All 0s if None.
        framework: An optional framework hint (to figure out,
            e.g. whether to transpose torch weight matrices).

    Returns:
        The dense layer's output.
    """

    def map_(data, transpose=False):
        if torch:
            if isinstance(data, torch.Tensor):
                data = data.cpu().detach().numpy()
        if tf and tf.executing_eagerly():
            if isinstance(data, tf.Variable):
                data = data.numpy()
        if transpose:
            data = np.transpose(data)
        return data
    x = map_(x)
    transpose = framework == 'torch' and (x.shape[1] != weights.shape[0] and x.shape[1] == weights.shape[1])
    weights = map_(weights, transpose=transpose)
    biases = map_(biases)
    return np.matmul(x, weights) + (0.0 if biases is None else biases)
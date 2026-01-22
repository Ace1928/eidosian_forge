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
Returns the softmax values for x.

    The exact formula used is:
    S(xi) = e^xi / SUMj(e^xj), where j goes over all elements in x.

    Args:
        x: The input to the softmax function.
        axis: The axis along which to softmax.
        epsilon: Optional epsilon as a minimum value. If None, use
            `SMALL_NUMBER`.

    Returns:
        The softmax over x.
    
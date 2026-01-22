import logging
import os
import warnings
from typing import Dict, List, Optional, TYPE_CHECKING, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Discrete, MultiDiscrete
from packaging import version
import ray
from ray.rllib.models.repeated_values import RepeatedValues
from ray.rllib.utils.annotations import Deprecated, PublicAPI, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import SMALL_NUMBER
from ray.rllib.utils.typing import (
@PublicAPI
def convert_to_torch_tensor(x: TensorStructType, device: Optional[str]=None):
    """Converts any struct to torch.Tensors.

    x: Any (possibly nested) struct, the values in which will be
        converted and returned as a new struct with all leaves converted
        to torch tensors.

    Returns:
        Any: A new struct with the same structure as `x`, but with all
            values converted to torch Tensor types. This does not convert possibly
            nested elements that are None because torch has no representation for that.
    """

    def mapping(item):
        if item is None:
            return item
        if isinstance(item, RepeatedValues):
            return RepeatedValues(tree.map_structure(mapping, item.values), item.lengths, item.max_len)
        if torch.is_tensor(item):
            tensor = item
        elif isinstance(item, np.ndarray):
            if item.dtype == object or item.dtype.type is np.str_:
                return item
            elif item.flags.writeable is False:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    tensor = torch.from_numpy(item)
            else:
                tensor = torch.from_numpy(item)
        else:
            tensor = torch.from_numpy(np.asarray(item))
        if tensor.is_floating_point():
            tensor = tensor.float()
        return tensor if device is None else tensor.to(device)
    return tree.map_structure(mapping, x)
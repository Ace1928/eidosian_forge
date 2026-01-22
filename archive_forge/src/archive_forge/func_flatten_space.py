import gymnasium as gym
from gymnasium.spaces import Tuple, Dict
import numpy as np
from ray.rllib.utils.annotations import DeveloperAPI
import tree  # pip install dm_tree
from typing import Any, List, Optional, Union
@DeveloperAPI
def flatten_space(space: gym.Space) -> List[gym.Space]:
    """Flattens a gym.Space into its primitive components.

    Primitive components are any non Tuple/Dict spaces.

    Args:
        space: The gym.Space to flatten. This may be any
            supported type (including nested Tuples and Dicts).

    Returns:
        List[gym.Space]: The flattened list of primitive Spaces. This list
            does not contain Tuples or Dicts anymore.
    """

    def _helper_flatten(space_, return_list):
        from ray.rllib.utils.spaces.flexdict import FlexDict
        if isinstance(space_, Tuple):
            for s in space_:
                _helper_flatten(s, return_list)
        elif isinstance(space_, (Dict, FlexDict)):
            for k in sorted(space_.spaces):
                _helper_flatten(space_[k], return_list)
        else:
            return_list.append(space_)
    ret = []
    _helper_flatten(space, ret)
    return ret
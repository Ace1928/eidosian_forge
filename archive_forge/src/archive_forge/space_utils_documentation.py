import gymnasium as gym
from gymnasium.spaces import Tuple, Dict
import numpy as np
from ray.rllib.utils.annotations import DeveloperAPI
import tree  # pip install dm_tree
from typing import Any, List, Optional, Union
Convert all the components of the element to match the space dtypes.

    Args:
        element: The element to be converted.
        sampled_element: An element sampled from a space to be matched
            to.

    Returns:
        The input element, but with all its components converted to match
        the space dtypes.
    
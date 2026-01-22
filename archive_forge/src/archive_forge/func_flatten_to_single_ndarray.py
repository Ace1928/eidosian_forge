import gymnasium as gym
from gymnasium.spaces import Tuple, Dict
import numpy as np
from ray.rllib.utils.annotations import DeveloperAPI
import tree  # pip install dm_tree
from typing import Any, List, Optional, Union
@DeveloperAPI
def flatten_to_single_ndarray(input_):
    """Returns a single np.ndarray given a list/tuple of np.ndarrays.

    Args:
        input_ (Union[List[np.ndarray], np.ndarray]): The list of ndarrays or
            a single ndarray.

    Returns:
        np.ndarray: The result after concatenating all single arrays in input_.

    .. testcode::
        :skipif: True

        flatten_to_single_ndarray([
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            np.array([7, 8, 9]),
        ])

    .. testoutput::

        np.array([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
        ])
    """
    if isinstance(input_, (list, tuple, dict)):
        expanded = []
        for in_ in tree.flatten(input_):
            expanded.append(np.reshape(in_, [-1]))
        input_ = np.concatenate(expanded, axis=0).flatten()
    return input_
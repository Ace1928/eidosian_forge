import inspect
from copy import deepcopy
import numpy as np
import gym
from gym import logger, spaces
from gym.utils.passive_env_checker import (
def data_equivalence(data_1, data_2) -> bool:
    """Assert equality between data 1 and 2, i.e observations, actions, info.

    Args:
        data_1: data structure 1
        data_2: data structure 2

    Returns:
        If observation 1 and 2 are equivalent
    """
    if type(data_1) == type(data_2):
        if isinstance(data_1, dict):
            return data_1.keys() == data_2.keys() and all((data_equivalence(data_1[k], data_2[k]) for k in data_1.keys()))
        elif isinstance(data_1, (tuple, list)):
            return len(data_1) == len(data_2) and all((data_equivalence(o_1, o_2) for o_1, o_2 in zip(data_1, data_2)))
        elif isinstance(data_1, np.ndarray):
            return data_1.shape == data_2.shape and np.allclose(data_1, data_2, atol=1e-05)
        else:
            return data_1 == data_2
    else:
        return False
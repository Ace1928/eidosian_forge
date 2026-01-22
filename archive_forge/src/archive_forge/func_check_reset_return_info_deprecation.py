import inspect
from copy import deepcopy
import numpy as np
import gym
from gym import logger, spaces
from gym.utils.passive_env_checker import (
def check_reset_return_info_deprecation(env: gym.Env):
    """Makes sure support for deprecated `return_info` argument is dropped.

    Args:
        env: The environment to check
    Raises:
        UserWarning
    """
    signature = inspect.signature(env.reset)
    if 'return_info' in signature.parameters:
        logger.warn('`return_info` is deprecated as an optional argument to `reset`. `reset`should now always return `obs, info` where `obs` is an observation, and `info` is a dictionarycontaining additional information.')
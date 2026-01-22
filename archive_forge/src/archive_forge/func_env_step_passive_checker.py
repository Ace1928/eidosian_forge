import inspect
from functools import partial
from typing import Callable
import numpy as np
from gym import Space, error, logger, spaces
def env_step_passive_checker(env, action):
    """A passive check for the environment step, investigating the returning data then returning the data unchanged."""
    result = env.step(action)
    assert isinstance(result, tuple), f'Expects step result to be a tuple, actual type: {type(result)}'
    if len(result) == 4:
        logger.deprecation('Core environment is written in old step API which returns one bool instead of two. It is recommended to rewrite the environment with new step API. ')
        obs, reward, done, info = result
        if not isinstance(done, (bool, np.bool8)):
            logger.warn(f'Expects `done` signal to be a boolean, actual type: {type(done)}')
    elif len(result) == 5:
        obs, reward, terminated, truncated, info = result
        if not isinstance(terminated, (bool, np.bool8)):
            logger.warn(f'Expects `terminated` signal to be a boolean, actual type: {type(terminated)}')
        if not isinstance(truncated, (bool, np.bool8)):
            logger.warn(f'Expects `truncated` signal to be a boolean, actual type: {type(truncated)}')
    else:
        raise error.Error(f'Expected `Env.step` to return a four or five element tuple, actual number of elements returned: {len(result)}.')
    check_obs(obs, env.observation_space, 'step')
    if not (np.issubdtype(type(reward), np.integer) or np.issubdtype(type(reward), np.floating)):
        logger.warn(f'The reward returned by `step()` must be a float, int, np.integer or np.floating, actual type: {type(reward)}')
    else:
        if np.isnan(reward):
            logger.warn('The reward is a NaN value.')
        if np.isinf(reward):
            logger.warn('The reward is an inf value.')
    assert isinstance(info, dict), f'The `info` returned by `step()` must be a python dictionary, actual type: {type(info)}'
    return result
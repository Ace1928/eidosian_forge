import inspect
from functools import partial
from typing import Callable
import numpy as np
from gym import Space, error, logger, spaces
def env_reset_passive_checker(env, **kwargs):
    """A passive check of the `Env.reset` function investigating the returning reset information and returning the data unchanged."""
    signature = inspect.signature(env.reset)
    if 'seed' not in signature.parameters and 'kwargs' not in signature.parameters:
        logger.warn('Future gym versions will require that `Env.reset` can be passed a `seed` instead of using `Env.seed` for resetting the environment random number generator.')
    else:
        seed_param = signature.parameters.get('seed')
        if seed_param is not None and seed_param.default is not None:
            logger.warn(f'The default seed argument in `Env.reset` should be `None`, otherwise the environment will by default always be deterministic. Actual default: {seed_param}')
    if 'options' not in signature.parameters and 'kwargs' not in signature.parameters:
        logger.warn('Future gym versions will require that `Env.reset` can be passed `options` to allow the environment initialisation to be passed additional information.')
    result = env.reset(**kwargs)
    if not isinstance(result, tuple):
        logger.warn(f'The result returned by `env.reset()` was not a tuple of the form `(obs, info)`, where `obs` is a observation and `info` is a dictionary containing additional information. Actual type: `{type(result)}`')
    elif len(result) != 2:
        logger.warn('The result returned by `env.reset()` should be `(obs, info)` by default, , where `obs` is a observation and `info` is a dictionary containing additional information.')
    else:
        obs, info = result
        check_obs(obs, env.observation_space, 'reset')
        assert isinstance(info, dict), f'The second element returned by `env.reset()` was not a dictionary, actual type: {type(info)}'
    return result
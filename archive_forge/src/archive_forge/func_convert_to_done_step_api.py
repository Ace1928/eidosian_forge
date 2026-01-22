from typing import Tuple, Union
import numpy as np
from gym.core import ObsType
def convert_to_done_step_api(step_returns: Union[TerminatedTruncatedStepType, DoneStepType], is_vector_env: bool=False) -> DoneStepType:
    """Function to transform step returns to old step API irrespective of input API.

    Args:
        step_returns (tuple): Items returned by step(). Can be (obs, rew, done, info) or (obs, rew, terminated, truncated, info)
        is_vector_env (bool): Whether the step_returns are from a vector environment
    """
    if len(step_returns) == 4:
        return step_returns
    else:
        assert len(step_returns) == 5
        observations, rewards, terminated, truncated, infos = step_returns
        if is_vector_env is False:
            if truncated or terminated:
                infos['TimeLimit.truncated'] = truncated and (not terminated)
            return (observations, rewards, terminated or truncated, infos)
        elif isinstance(infos, list):
            for info, env_truncated, env_terminated in zip(infos, truncated, terminated):
                if env_truncated or env_terminated:
                    info['TimeLimit.truncated'] = env_truncated and (not env_terminated)
            return (observations, rewards, np.logical_or(terminated, truncated), infos)
        elif isinstance(infos, dict):
            if np.logical_or(np.any(truncated), np.any(terminated)):
                infos['TimeLimit.truncated'] = np.logical_and(truncated, np.logical_not(terminated))
            return (observations, rewards, np.logical_or(terminated, truncated), infos)
        else:
            raise TypeError(f'Unexpected value of infos, as is_vector_envs=False, expects `info` to be a list or dict, actual type: {type(infos)}')
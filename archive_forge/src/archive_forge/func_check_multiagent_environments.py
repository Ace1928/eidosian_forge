import logging
import traceback
from copy import copy
from typing import TYPE_CHECKING, Optional, Set, Union
import numpy as np
import tree  # pip install dm_tree
from ray.actor import ActorHandle
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.error import ERR_MSG_OLD_GYM_API, UnsupportedSpaceException
from ray.rllib.utils.gym import check_old_gym_env, try_import_gymnasium_and_gym
from ray.rllib.utils.spaces.space_utils import (
from ray.rllib.utils.typing import EnvType
from ray.util import log_once
@DeveloperAPI
def check_multiagent_environments(env: 'MultiAgentEnv') -> None:
    """Checking for common errors in RLlib MultiAgentEnvs.

    Args:
        env: The env to be checked.
    """
    from ray.rllib.env import MultiAgentEnv
    if not isinstance(env, MultiAgentEnv):
        raise ValueError('The passed env is not a MultiAgentEnv.')
    elif not (hasattr(env, 'observation_space') and hasattr(env, 'action_space') and hasattr(env, '_agent_ids') and hasattr(env, '_obs_space_in_preferred_format') and hasattr(env, '_action_space_in_preferred_format')):
        if log_once('ma_env_super_ctor_called'):
            logger.warning(f"Your MultiAgentEnv {env} does not have some or all of the needed base-class attributes! Make sure you call `super().__init__()` from within your MutiAgentEnv's constructor. This will raise an error in the future.")
        return
    try:
        obs_and_infos = env.reset(seed=42, options={})
        check_old_gym_env(reset_results=obs_and_infos)
    except Exception as e:
        raise ValueError(ERR_MSG_OLD_GYM_API.format(env, 'In particular, the `reset()` method seems to be faulty.')) from e
    reset_obs, reset_infos = obs_and_infos
    sampled_obs = env.observation_space_sample()
    _check_if_element_multi_agent_dict(env, reset_obs, 'reset()')
    _check_if_element_multi_agent_dict(env, sampled_obs, 'env.observation_space_sample()')
    try:
        env.observation_space_contains(reset_obs)
    except Exception as e:
        raise ValueError('Your observation_space_contains function has some error ') from e
    if not env.observation_space_contains(reset_obs):
        error = _not_contained_error('env.reset', 'observation') + f'\n\n reset_obs: {reset_obs}\n\n env.observation_space_sample(): {sampled_obs}\n\n '
        raise ValueError(error)
    if not env.observation_space_contains(sampled_obs):
        error = _not_contained_error('observation_space_sample', 'observation') + f'\n\n env.observation_space_sample(): {sampled_obs}\n\n '
        raise ValueError(error)
    sampled_action = env.action_space_sample(list(reset_obs.keys()))
    _check_if_element_multi_agent_dict(env, sampled_action, 'action_space_sample')
    try:
        env.action_space_contains(sampled_action)
    except Exception as e:
        raise ValueError('Your action_space_contains function has some error ') from e
    if not env.action_space_contains(sampled_action):
        error = _not_contained_error('action_space_sample', 'action') + f'\n\n sampled_action {sampled_action}\n\n'
        raise ValueError(error)
    try:
        results = env.step(sampled_action)
        check_old_gym_env(step_results=results)
    except Exception as e:
        raise ValueError(ERR_MSG_OLD_GYM_API.format(env, 'In particular, the `step()` method seems to be faulty.')) from e
    next_obs, reward, done, truncated, info = results
    _check_if_element_multi_agent_dict(env, next_obs, 'step, next_obs')
    _check_if_element_multi_agent_dict(env, reward, 'step, reward')
    _check_if_element_multi_agent_dict(env, done, 'step, done')
    _check_if_element_multi_agent_dict(env, truncated, 'step, truncated')
    _check_if_element_multi_agent_dict(env, info, 'step, info')
    _check_reward({'dummy_env_id': reward}, base_env=True, agent_ids=env.get_agent_ids())
    _check_done_and_truncated({'dummy_env_id': done}, {'dummy_env_id': truncated}, base_env=True, agent_ids=env.get_agent_ids())
    _check_info({'dummy_env_id': info}, base_env=True, agent_ids=env.get_agent_ids())
    if not env.observation_space_contains(next_obs):
        error = _not_contained_error('env.step(sampled_action)', 'observation') + f':\n\n next_obs: {next_obs} \n\n sampled_obs: {sampled_obs}'
        raise ValueError(error)
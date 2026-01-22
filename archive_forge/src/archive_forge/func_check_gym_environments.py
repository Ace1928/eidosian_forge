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
def check_gym_environments(env: Union[gym.Env, 'old_gym.Env'], config: 'AlgorithmConfig') -> None:
    """Checking for common errors in a gymnasium/gym environments.

    Args:
        env: Environment to be checked.
        config: Additional checks config.

    Warning:
        If env has no attribute spec with a sub attribute,
            max_episode_steps.

    Raises:
        AttributeError: If env has no observation space.
        AttributeError: If env has no action space.
        ValueError: Observation space must be a gym.spaces.Space.
        ValueError: Action space must be a gym.spaces.Space.
        ValueError: Observation sampled from observation space must be
            contained in the observation space.
        ValueError: Action sampled from action space must be
            contained in the observation space.
        ValueError: If env cannot be resetted.
        ValueError: If an observation collected from a call to env.reset().
            is not contained in the observation_space.
        ValueError: If env cannot be stepped via a call to env.step().
        ValueError: If the observation collected from env.step() is not
            contained in the observation_space.
        AssertionError: If env.step() returns a reward that is not an
            int or float.
        AssertionError: IF env.step() returns a done that is not a bool.
        AssertionError: If env.step() returns an env_info that is not a dict.
    """
    if old_gym and isinstance(env, old_gym.Env):
        raise ValueError(ERR_MSG_OLD_GYM_API.format(env, ''))
    if not hasattr(env, 'observation_space'):
        raise AttributeError('Env must have observation_space.')
    if not hasattr(env, 'action_space'):
        raise AttributeError('Env must have action_space.')
    if not isinstance(env.observation_space, gym.spaces.Space):
        raise ValueError('Observation space must be a gymnasium.Space!')
    if not isinstance(env.action_space, gym.spaces.Space):
        raise ValueError('Action space must be a gymnasium.Space!')
    if not hasattr(env, 'spec') or not hasattr(env.spec, 'max_episode_steps'):
        if log_once('max_episode_steps'):
            logger.warning("Your env doesn't have a .spec.max_episode_steps attribute. Your horizon will default to infinity, and your environment will not be reset.")
    sampled_observation = env.observation_space.sample()
    sampled_action = env.action_space.sample()
    try:
        env.reset()
    except Exception as e:
        raise ValueError("Your gymnasium.Env's `reset()` method raised an Exception!") from e
    try:
        obs_and_infos = env.reset(seed=None, options={})
        check_old_gym_env(reset_results=obs_and_infos)
    except Exception as e:
        raise ValueError(ERR_MSG_OLD_GYM_API.format(env, 'In particular, the `reset()` method seems to be faulty.')) from e
    reset_obs, reset_infos = obs_and_infos
    if not env.observation_space.contains(reset_obs):
        temp_sampled_reset_obs = convert_element_to_space_type(reset_obs, sampled_observation)
        if not env.observation_space.contains(temp_sampled_reset_obs):
            key, space, space_type, value, value_type = _find_offending_sub_space(env.observation_space, temp_sampled_reset_obs)
            raise ValueError("The observation collected from env.reset() was not contained within your env's observation space. It is possible that there was a type mismatch, or that one of the sub-observations was out of bounds:\n {}(sub-)obs: {} ({})\n (sub-)observation space: {} ({})".format("path: '" + key + "'\n " if key else '', value, value_type, space, space_type))
    if isinstance(reset_obs, dict):
        if config.action_mask_key in reset_obs:
            sampled_action = env.action_space.sample(mask=reset_obs[config.action_mask_key])
    try:
        results = env.step(sampled_action)
    except Exception as e:
        raise ValueError("Your gymnasium.Env's `step()` method raised an Exception!") from e
    try:
        check_old_gym_env(step_results=results)
    except Exception as e:
        raise ValueError(ERR_MSG_OLD_GYM_API.format(env, 'In particular, the `step()` method seems to be faulty.')) from e
    next_obs, reward, done, truncated, info = results
    if not env.observation_space.contains(next_obs):
        temp_sampled_next_obs = convert_element_to_space_type(next_obs, sampled_observation)
        if not env.observation_space.contains(temp_sampled_next_obs):
            key, space, space_type, value, value_type = _find_offending_sub_space(env.observation_space, temp_sampled_next_obs)
            error = "The observation collected from env.step(sampled_action) was not contained within your env's observation space. It is possible that there was a type mismatch, or that one of the sub-observations was out of bounds: \n\n {}(sub-)obs: {} ({})\n (sub-)observation space: {} ({})".format("path='" + key + "'\n " if key else '', value, value_type, space, space_type)
            raise ValueError(error)
    _check_done_and_truncated(done, truncated)
    _check_reward(reward)
    _check_info(info)
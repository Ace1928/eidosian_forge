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
def check_env(env: EnvType, config: Optional['AlgorithmConfig']=None) -> None:
    """Run pre-checks on env that uncover common errors in environments.

    Args:
        env: Environment to be checked.
        config: Additional checks config.

    Raises:
        ValueError: If env is not an instance of SUPPORTED_ENVIRONMENT_TYPES.
        ValueError: See check_gym_env docstring for details.
    """
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.env import BaseEnv, ExternalEnv, ExternalMultiAgentEnv, MultiAgentEnv, RemoteBaseEnv, VectorEnv
    if hasattr(env, '_skip_env_checking') and env._skip_env_checking:
        if log_once('skip_env_checking'):
            logger.warning('Skipping env checking for this experiment')
        return
    try:
        if not isinstance(env, (BaseEnv, gym.Env, MultiAgentEnv, RemoteBaseEnv, VectorEnv, ExternalMultiAgentEnv, ExternalEnv, ActorHandle)) and (not old_gym or not isinstance(env, old_gym.Env)):
            raise ValueError(f'Env must be of one of the following supported types: BaseEnv, gymnasium.Env, gym.Env, MultiAgentEnv, VectorEnv, RemoteBaseEnv, ExternalMultiAgentEnv, ExternalEnv, but instead is of type {type(env)}.')
        if isinstance(env, MultiAgentEnv):
            check_multiagent_environments(env)
        elif isinstance(env, VectorEnv):
            check_vector_env(env)
        elif isinstance(env, gym.Env) or (old_gym and isinstance(env, old_gym.Env)):
            check_gym_environments(env, AlgorithmConfig() if config is None else config)
        elif isinstance(env, BaseEnv):
            check_base_env(env)
        else:
            logger.warning("Env checking isn't implemented for RemoteBaseEnvs, ExternalMultiAgentEnv, ExternalEnvs or environments that are Ray actors.")
    except Exception:
        actual_error = traceback.format_exc()
        raise ValueError(f"{actual_error}\nThe above error has been found in your environment! We've added a module for checking your custom environments. It may cause your experiment to fail if your environment is not set up correctly. You can disable this behavior via calling `config.environment(disable_env_checking=True)`. You can run the environment checking module standalone by calling ray.rllib.utils.check_env([your env]).")
import gymnasium as gym
import logging
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import (
from ray.rllib.utils.typing import (
from ray.util import log_once
@PublicAPI
def to_base_env(self, make_env: Optional[Callable[[int], EnvType]]=None, num_envs: int=1, remote_envs: bool=False, remote_env_batch_wait_ms: int=0, restart_failed_sub_environments: bool=False) -> 'BaseEnv':
    """Converts an RLlib MultiAgentEnv into a BaseEnv object.

        The resulting BaseEnv is always vectorized (contains n
        sub-environments) to support batched forward passes, where n may
        also be 1. BaseEnv also supports async execution via the `poll` and
        `send_actions` methods and thus supports external simulators.

        Args:
            make_env: A callable taking an int as input (which indicates
                the number of individual sub-environments within the final
                vectorized BaseEnv) and returning one individual
                sub-environment.
            num_envs: The number of sub-environments to create in the
                resulting (vectorized) BaseEnv. The already existing `env`
                will be one of the `num_envs`.
            remote_envs: Whether each sub-env should be a @ray.remote
                actor. You can set this behavior in your config via the
                `remote_worker_envs=True` option.
            remote_env_batch_wait_ms: The wait time (in ms) to poll remote
                sub-environments for, if applicable. Only used if
                `remote_envs` is True.
            restart_failed_sub_environments: If True and any sub-environment (within
                a vectorized env) throws any error during env stepping, we will try to
                restart the faulty sub-environment. This is done
                without disturbing the other (still intact) sub-environments.

        Returns:
            The resulting BaseEnv object.
        """
    from ray.rllib.env.remote_base_env import RemoteBaseEnv
    if remote_envs:
        env = RemoteBaseEnv(make_env, num_envs, multiagent=True, remote_env_batch_wait_ms=remote_env_batch_wait_ms, restart_failed_sub_environments=restart_failed_sub_environments)
    else:
        env = MultiAgentEnvWrapper(make_env=make_env, existing_envs=[self], num_envs=num_envs, restart_failed_sub_environments=restart_failed_sub_environments)
    return env
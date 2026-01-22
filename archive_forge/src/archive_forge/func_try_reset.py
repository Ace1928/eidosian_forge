import gymnasium as gym
import logging
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import (
from ray.rllib.utils.typing import (
from ray.util import log_once
@override(BaseEnv)
def try_reset(self, env_id: Optional[EnvID]=None, *, seed: Optional[int]=None, options: Optional[dict]=None) -> Optional[Tuple[MultiEnvDict, MultiEnvDict]]:
    ret_obs = {}
    ret_infos = {}
    if isinstance(env_id, int):
        env_id = [env_id]
    if env_id is None:
        env_id = list(range(len(self.envs)))
    for idx in env_id:
        obs, infos = self.env_states[idx].reset(seed=seed, options=options)
        if isinstance(obs, Exception):
            if self.restart_failed_sub_environments:
                self.env_states[idx].env = self.envs[idx] = self.make_env(idx)
            else:
                raise obs
        else:
            assert isinstance(obs, dict), 'Not a multi-agent obs dict!'
        if obs is not None:
            if idx in self.terminateds:
                self.terminateds.remove(idx)
            if idx in self.truncateds:
                self.truncateds.remove(idx)
        ret_obs[idx] = obs
        ret_infos[idx] = infos
    return (ret_obs, ret_infos)
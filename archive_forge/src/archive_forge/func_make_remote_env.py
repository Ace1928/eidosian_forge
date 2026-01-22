import gymnasium as gym
import logging
from typing import Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import ray
from ray.util import log_once
from ray.rllib.env.base_env import BaseEnv, _DUMMY_AGENT_ID, ASYNC_RESET_RETURN
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import AgentID, EnvID, EnvType, MultiEnvDict
def make_remote_env(i):
    logger.info('Launching env {} in remote actor'.format(i))
    if self.multiagent:
        sub_env = _RemoteMultiAgentEnv.remote(self.make_env, i)
    else:
        sub_env = _RemoteSingleAgentEnv.remote(self.make_env, i)
    if self.worker is not None:
        self.worker.callbacks.on_sub_environment_created(worker=self.worker, sub_environment=sub_env, env_context=self.worker.env_context.copy_with_overrides(vector_index=i))
    return sub_env
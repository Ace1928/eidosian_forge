import functools
import gymnasium as gym
import logging
import importlib.util
import os
from typing import (
import ray
from ray.actor import ActorHandle
from ray.exceptions import RayActorError
from ray.rllib.core.learner import LearnerGroup
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.utils.actor_manager import RemoteCallResults
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.offline import get_dataset_and_shards
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.actor_manager import FaultTolerantActorManager
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.typing import (
@DeveloperAPI
def foreach_worker_with_id(self, func: Callable[[int, EnvRunner], T], *, local_worker: bool=True, healthy_only: bool=False, remote_worker_ids: List[int]=None, timeout_seconds: Optional[int]=None) -> List[T]:
    """Similar to foreach_worker(), but calls the function with id of the worker too.

        Args:
            func: The function to call for each worker (as only arg).
            local_worker: Whether apply `func` on local worker too. Default is True.
            healthy_only: Apply `func` on known-to-be healthy workers only.
            remote_worker_ids: Apply `func` on a selected set of remote workers.
            timeout_seconds: Time to wait for results. Default is None.

        Returns:
             The list of return values of all calls to `func([worker, id])`.
        """
    local_result = []
    if local_worker and self.local_worker() is not None:
        local_result = [func(0, self.local_worker())]
    if not remote_worker_ids:
        remote_worker_ids = self.__worker_manager.actor_ids()
    funcs = [functools.partial(func, i) for i in remote_worker_ids]
    remote_results = self.__worker_manager.foreach_actor(funcs, healthy_only=healthy_only, remote_actor_ids=remote_worker_ids, timeout_seconds=timeout_seconds)
    handle_remote_call_result_errors(remote_results, self._ignore_worker_failures)
    remote_results = [r.get() for r in remote_results.ignore_errors()]
    return local_result + remote_results
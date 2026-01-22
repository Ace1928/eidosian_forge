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
def foreach_worker(self, func: Callable[[EnvRunner], T], *, local_worker: bool=True, healthy_only: bool=False, remote_worker_ids: List[int]=None, timeout_seconds: Optional[int]=None, return_obj_refs: bool=False, mark_healthy: bool=False) -> List[T]:
    """Calls the given function with each worker instance as the argument.

        Args:
            func: The function to call for each worker (as only arg).
            local_worker: Whether apply `func` on local worker too. Default is True.
            healthy_only: Apply `func` on known-to-be healthy workers only.
            remote_worker_ids: Apply `func` on a selected set of remote workers.
            timeout_seconds: Time to wait for results. Default is None.
            return_obj_refs: whether to return ObjectRef instead of actual results.
                Note, for fault tolerance reasons, these returned ObjectRefs should
                never be resolved with ray.get() outside of this WorkerSet.
            mark_healthy: Whether to mark the worker as healthy based on call results.

        Returns:
             The list of return values of all calls to `func([worker])`.
        """
    assert not return_obj_refs or not local_worker, 'Can not return ObjectRef from local worker.'
    local_result = []
    if local_worker and self.local_worker() is not None:
        local_result = [func(self.local_worker())]
    if not self.__worker_manager.actor_ids():
        return local_result
    remote_results = self.__worker_manager.foreach_actor(func, healthy_only=healthy_only, remote_actor_ids=remote_worker_ids, timeout_seconds=timeout_seconds, return_obj_refs=return_obj_refs, mark_healthy=mark_healthy)
    handle_remote_call_result_errors(remote_results, self._ignore_worker_failures)
    remote_results = [r.get() for r in remote_results.ignore_errors()]
    return local_result + remote_results
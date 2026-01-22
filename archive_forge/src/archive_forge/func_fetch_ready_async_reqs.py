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
def fetch_ready_async_reqs(self, *, timeout_seconds: Optional[int]=0, return_obj_refs: bool=False, mark_healthy: bool=False) -> List[Tuple[int, T]]:
    """Get esults from outstanding asynchronous requests that are ready.

        Args:
            timeout_seconds: Time to wait for results. Default is 0, meaning
                those requests that are already ready.
            return_obj_refs: Whether to return ObjectRef instead of actual results.
            mark_healthy: Whether to mark the worker as healthy based on call results.

        Returns:
            A list of results successfully returned from outstanding remote calls,
            paired with the indices of the callee workers.
        """
    remote_results = self.__worker_manager.fetch_ready_async_reqs(timeout_seconds=timeout_seconds, return_obj_refs=return_obj_refs, mark_healthy=mark_healthy)
    handle_remote_call_result_errors(remote_results, self._ignore_worker_failures)
    return [(r.actor_id, r.get()) for r in remote_results.ignore_errors()]
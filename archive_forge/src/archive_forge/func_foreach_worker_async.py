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
def foreach_worker_async(self, func: Callable[[EnvRunner], T], *, healthy_only: bool=False, remote_worker_ids: List[int]=None) -> int:
    """Calls the given function asynchronously with each worker as the argument.

        foreach_worker_async() does not return results directly. Instead,
        fetch_ready_async_reqs() can be used to pull results in an async manner
        whenever they are available.

        Args:
            func: The function to call for each worker (as only arg).
            healthy_only: Apply `func` on known-to-be healthy workers only.
            remote_worker_ids: Apply `func` on a selected set of remote workers.

        Returns:
             The number of async requests that are currently in-flight.
        """
    return self.__worker_manager.foreach_actor_async(func, healthy_only=healthy_only, remote_actor_ids=remote_worker_ids)
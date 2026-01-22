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
def sync_weights(self, policies: Optional[List[PolicyID]]=None, from_worker_or_learner_group: Optional[Union[EnvRunner, LearnerGroup]]=None, to_worker_indices: Optional[List[int]]=None, global_vars: Optional[Dict[str, TensorType]]=None, timeout_seconds: Optional[int]=0) -> None:
    """Syncs model weights from the given weight source to all remote workers.

        Weight source can be either a (local) rollout worker or a learner_group. It
        should just implement a `get_weights` method.

        Args:
            policies: Optional list of PolicyIDs to sync weights for.
                If None (default), sync weights to/from all policies.
            from_worker_or_learner_group: Optional (local) EnvRunner instance or
                LearnerGroup instance to sync from. If None (default),
                sync from this WorkerSet's local worker.
            to_worker_indices: Optional list of worker indices to sync the
                weights to. If None (default), sync to all remote workers.
            global_vars: An optional global vars dict to set this
                worker to. If None, do not update the global_vars.
            timeout_seconds: Timeout in seconds to wait for the sync weights
                calls to complete. Default is 0 (sync-and-forget, do not wait
                for any sync calls to finish). This significantly improves
                algorithm performance.
        """
    if self.local_worker() is None and from_worker_or_learner_group is None:
        raise TypeError('No `local_worker` in WorkerSet, must provide `from_worker_or_learner_group` arg in `sync_weights()`!')
    weights = None
    if self.num_remote_workers() or from_worker_or_learner_group is not None:
        weights_src = from_worker_or_learner_group or self.local_worker()
        if weights_src is None:
            raise ValueError('`from_worker_or_trainer` is None. In this case, workerset should have local_worker. But local_worker is also None.')
        weights = weights_src.get_weights(policies)

        def set_weight(w):
            w.set_weights(weights, global_vars)
        self.foreach_worker(func=set_weight, local_worker=False, remote_worker_ids=to_worker_indices, healthy_only=True, timeout_seconds=timeout_seconds)
    if self.local_worker() is not None:
        if from_worker_or_learner_group is not None:
            self.local_worker().set_weights(weights, global_vars=global_vars)
        elif global_vars is not None:
            self.local_worker().set_global_vars(global_vars)
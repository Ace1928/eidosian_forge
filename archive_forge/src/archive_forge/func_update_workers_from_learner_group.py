import copy
import dataclasses
from functools import partial
import logging
import platform
import queue
import random
from typing import Callable, List, Optional, Set, Tuple, Type, Union
import numpy as np
import tree  # pip install dm_tree
import ray
from ray import ObjectRef
from ray.rllib import SampleBatch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.impala.impala_learner import (
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.evaluation.worker_set import handle_remote_call_result_errors
from ray.rllib.execution.buffers.mixin_replay_buffer import MixInMultiAgentReplayBuffer
from ray.rllib.execution.learner_thread import LearnerThread
from ray.rllib.execution.multi_gpu_learner_thread import MultiGPULearnerThread
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import concat_samples
from ray.rllib.utils.actor_manager import (
from ray.rllib.utils.actors import create_colocated_actors
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import ALL_MODULES
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import ReplayMode
from ray.rllib.utils.replay_buffers.replay_buffer import _ALL_POLICIES
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.typing import (
from ray.tune.execution.placement_groups import PlacementGroupFactory
def update_workers_from_learner_group(self, workers_that_need_updates: Set[int], policy_ids: Optional[List[PolicyID]]=None):
    """Updates all RolloutWorkers that require updating.

        Updates only if NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS has been
        reached and the worker has sent samples in this iteration. Also only updates
        those policies, whose IDs are given via `policies` (if None, update all
        policies).

        Args:
            workers_that_need_updates: Set of worker IDs that need to be updated.
            policy_ids: Optional list of Policy IDs to update. If None, will update all
                policies on the to-be-updated workers.
        """
    self._counters[NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS] += 1
    if self._counters[NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS] >= self.config.broadcast_interval and workers_that_need_updates:
        self._counters[NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS] = 0
        self._counters[NUM_SYNCH_WORKER_WEIGHTS] += 1
        weights = self.learner_group.get_weights(policy_ids)
        if self.config.num_rollout_workers == 0:
            worker = self.workers.local_worker()
            worker.set_weights(weights)
        else:
            weights_ref = ray.put(weights)
            self.workers.foreach_worker(func=lambda w: w.set_weights(ray.get(weights_ref)), local_worker=False, remote_worker_ids=list(workers_that_need_updates), timeout_seconds=0)
            if self.config.create_env_on_local_worker:
                self.workers.local_worker().set_weights(weights)
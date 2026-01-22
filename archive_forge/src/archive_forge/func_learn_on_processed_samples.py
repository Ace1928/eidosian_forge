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
def learn_on_processed_samples(self) -> ResultDict:
    """Update the learner group with the latest batch of processed samples.

        Returns:
            Aggregated results from the learner group after an update is completed.

        """
    if self.batches_to_place_on_learner:
        batches = self.batches_to_place_on_learner[:]
        self.batches_to_place_on_learner.clear()
        blocking = self.config.num_learner_workers == 0
        results = []
        for batch in batches:
            if blocking:
                result = self.learner_group.update(batch, reduce_fn=_reduce_impala_results, num_iters=self.config.num_sgd_iter, minibatch_size=self.config.minibatch_size)
                results = [result]
            else:
                results = self.learner_group.async_update(batch, reduce_fn=_reduce_impala_results, num_iters=self.config.num_sgd_iter, minibatch_size=self.config.minibatch_size)
            for r in results:
                self._counters[NUM_ENV_STEPS_TRAINED] += r[ALL_MODULES].pop(NUM_ENV_STEPS_TRAINED)
                self._counters[NUM_AGENT_STEPS_TRAINED] += r[ALL_MODULES].pop(NUM_AGENT_STEPS_TRAINED)
        self._counters.update(self.learner_group.get_in_queue_stats())
        if results:
            return tree.map_structure(lambda *x: np.mean(x), *results)
    return {}
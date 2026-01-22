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
def make_learner_thread(local_worker, config):
    if not config['simple_optimizer']:
        logger.info('Enabling multi-GPU mode, {} GPUs, {} parallel tower-stacks'.format(config['num_gpus'], config['num_multi_gpu_tower_stacks']))
        num_stacks = config['num_multi_gpu_tower_stacks']
        buffer_size = config['minibatch_buffer_size']
        if num_stacks < buffer_size:
            logger.warning(f'In multi-GPU mode you should have at least as many multi-GPU tower stacks (to load data into on one device) as you have stack-index slots in the buffer! You have configured {num_stacks} stacks and a buffer of size {buffer_size}. Setting `minibatch_buffer_size={num_stacks}`.')
            config['minibatch_buffer_size'] = num_stacks
        learner_thread = MultiGPULearnerThread(local_worker, num_gpus=config['num_gpus'], lr=config['lr'], train_batch_size=config['train_batch_size'], num_multi_gpu_tower_stacks=config['num_multi_gpu_tower_stacks'], num_sgd_iter=config['num_sgd_iter'], learner_queue_size=config['learner_queue_size'], learner_queue_timeout=config['learner_queue_timeout'])
    else:
        learner_thread = LearnerThread(local_worker, minibatch_buffer_size=config['minibatch_buffer_size'], num_sgd_iter=config['num_sgd_iter'], learner_queue_size=config['learner_queue_size'], learner_queue_timeout=config['learner_queue_timeout'])
    return learner_thread
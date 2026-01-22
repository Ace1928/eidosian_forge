import dataclasses
import logging
from typing import List, Optional, Type, Union, TYPE_CHECKING
import numpy as np
import tree
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo.ppo_learner import (
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.evaluation.postprocessing_v2 import postprocess_episodes_to_sample_batch
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.execution.rollout_ops import (
from ray.rllib.execution.train_ops import (
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.metrics import (
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.typing import ResultDict
from ray.util.debug import log_once
@override(AlgorithmConfig)
def get_learner_hyperparameters(self) -> PPOLearnerHyperparameters:
    base_hps = super().get_learner_hyperparameters()
    return PPOLearnerHyperparameters(use_critic=self.use_critic, use_kl_loss=self.use_kl_loss, kl_coeff=self.kl_coeff, kl_target=self.kl_target, vf_loss_coeff=self.vf_loss_coeff, entropy_coeff=self.entropy_coeff, entropy_coeff_schedule=self.entropy_coeff_schedule, clip_param=self.clip_param, vf_clip_param=self.vf_clip_param, **dataclasses.asdict(base_hps))
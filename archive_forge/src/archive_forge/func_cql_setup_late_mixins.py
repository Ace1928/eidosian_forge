import numpy as np
import gymnasium as gym
import logging
import tree
from typing import Dict, List, Tuple, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.sac.sac_tf_policy import (
from ray.rllib.algorithms.sac.sac_torch_policy import (
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_mixins import TargetNetworkMixin
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import LocalOptimizer, TensorType, AlgorithmConfigDict
from ray.rllib.utils.torch_utils import (
def cql_setup_late_mixins(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> None:
    setup_late_mixins(policy, obs_space, action_space, config)
    if config['lagrangian']:
        policy.model.log_alpha_prime = policy.model.log_alpha_prime.to(policy.device)
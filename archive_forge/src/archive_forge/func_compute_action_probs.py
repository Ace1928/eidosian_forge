import gymnasium as gym
import numpy as np
import tree
from typing import Dict, Any, List
import logging
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import convert_ma_batch_to_sample_batch
from ray.rllib.utils.policy import compute_log_likelihoods_from_input_dict
from ray.rllib.utils.annotations import (
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import TensorType, SampleBatchType
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
@ExperimentalAPI
def compute_action_probs(self, batch: SampleBatch):
    log_likelihoods = compute_log_likelihoods_from_input_dict(self.policy, batch)
    new_prob = np.exp(convert_to_numpy(log_likelihoods))
    if self.epsilon_greedy > 0.0:
        if not isinstance(self.policy.action_space, gym.spaces.Discrete):
            raise ValueError('Evaluation with epsilon-greedy exploration is only supported with discrete action spaces.')
        eps = self.epsilon_greedy
        new_prob = new_prob * (1 - eps) + eps / self.policy.action_space.n
    return new_prob
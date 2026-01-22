import logging
import numpy as np
import math
import pandas as pd
from typing import Dict, Any, Optional, List
from ray.data import Dataset
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, convert_ma_batch_to_sample_batch
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.offline.estimators.off_policy_estimator import OffPolicyEstimator
from ray.rllib.offline.estimators.fqe_torch_model import FQETorchModel
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
from ray.rllib.offline.offline_evaluation_utils import (
@override(OffPolicyEstimator)
def estimate_on_single_episode(self, episode: SampleBatch) -> Dict[str, Any]:
    estimates_per_epsiode = {}
    rewards, old_prob = (episode['rewards'], episode['action_prob'])
    new_prob = self.compute_action_probs(episode)
    weight = new_prob / old_prob
    v_behavior = 0.0
    v_target = 0.0
    q_values = self.model.estimate_q(episode)
    q_values = convert_to_numpy(q_values)
    v_values = self.model.estimate_v(episode)
    v_values = convert_to_numpy(v_values)
    assert q_values.shape == v_values.shape == (episode.count,)
    for t in reversed(range(episode.count)):
        v_behavior = rewards[t] + self.gamma * v_behavior
        v_target = v_values[t] + weight[t] * (rewards[t] + self.gamma * v_target - q_values[t])
    v_target = v_target.item()
    estimates_per_epsiode['v_behavior'] = v_behavior
    estimates_per_epsiode['v_target'] = v_target
    return estimates_per_epsiode
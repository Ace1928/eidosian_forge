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
def estimate_on_single_step_samples(self, batch: SampleBatch) -> Dict[str, List[float]]:
    estimates_per_epsiode = {}
    rewards, old_prob = (batch['rewards'], batch['action_prob'])
    new_prob = self.compute_action_probs(batch)
    q_values = self.model.estimate_q(batch)
    q_values = convert_to_numpy(q_values)
    v_values = self.model.estimate_v(batch)
    v_values = convert_to_numpy(v_values)
    v_behavior = rewards
    weight = new_prob / old_prob
    v_target = v_values + weight * (rewards - q_values)
    estimates_per_epsiode['v_behavior'] = v_behavior
    estimates_per_epsiode['v_target'] = v_target
    return estimates_per_epsiode
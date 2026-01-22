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
def compute_v_target(batch: pd.DataFrame, normalizer: float=1.0):
    weights = batch['weights'] / normalizer
    batch['v_target'] = batch['v_values'] + weights * (batch['rewards'] - batch['q_values'])
    batch['v_behavior'] = batch['rewards']
    return batch
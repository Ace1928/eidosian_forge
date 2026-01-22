from typing import Dict, Any, List
import numpy as np
import math
from ray.data import Dataset
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
from ray.rllib.offline.estimators.off_policy_estimator import OffPolicyEstimator
from ray.rllib.offline.offline_evaluation_utils import (
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.utils.annotations import override, DeveloperAPI
Computes the weighted importance sampling estimate on a dataset.

        Note: This estimate works for both continuous and discrete action spaces.

        Args:
            dataset: Dataset to compute the estimate on. Each record in dataset should
                include the following columns: `obs`, `actions`, `action_prob` and
                `rewards`. The `obs` on each row shoud be a vector of D dimensions.
            n_parallelism: Number of parallel workers to use for the computation.

        Returns:
            Dictionary with the following keys:
                v_target: The weighted importance sampling estimate.
                v_behavior: The behavior policy estimate.
                v_gain_mean: The mean of the gain of the target policy over the
                    behavior policy.
                v_gain_ste: The standard error of the gain of the target policy over
                    the behavior policy.
        
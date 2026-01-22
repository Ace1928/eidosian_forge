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
@OverrideToImplementCustomLogic
def peek_on_single_episode(self, episode: SampleBatch) -> None:
    """This is called on each episode before it is passed to
        estimate_on_single_episode(). Using this method, you can get a peek at the
        entire validation dataset before runnining the estimation. For examlpe if you
        need to perform any normalizations of any sorts on the dataset, you can compute
        the normalization parameters here.

        Args:
            episode: The episode that is split from the original batch. This is a
            sample batch type that is a single episode.
        """
    pass
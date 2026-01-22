import logging
import numpy as np
import random
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
@DeveloperAPI
def standardized(array: np.ndarray):
    """Normalize the values in an array.

    Args:
        array (np.ndarray): Array of values to normalize.

    Returns:
        array with zero mean and unit standard deviation.
    """
    return (array - array.mean()) / max(0.0001, array.std())
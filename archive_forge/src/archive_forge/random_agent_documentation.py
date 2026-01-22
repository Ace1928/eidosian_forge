import numpy as np
from typing import Optional
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, ALGO_DEPRECATION_WARNING
Sets the rollout configuration.

        Args:
            rollouts_per_iteration: How many episodes to run per training iteration.

        Returns:
            This updated AlgorithmConfig object.
        
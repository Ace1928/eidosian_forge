import abc
import os
import logging
from typing import Dict, Any
from ray.data import Dataset
from ray.rllib.policy import Policy
from ray.rllib.utils.annotations import DeveloperAPI, ExperimentalAPI
from ray.rllib.utils.typing import SampleBatchType
Calculates the estimate of the metrics based on the given offline dataset.

        Typically, the dataset is passed through only once via n_parallel tasks in
        mini-batches to improve the run-time of metric estimation.

        Args:
            dataset: The ray dataset object to do offline evaluation on.
            n_parallelism: The number of parallelism to use for the computation.

        Returns:
            Dict[str, Any]: A dictionary of the estimated values.
        
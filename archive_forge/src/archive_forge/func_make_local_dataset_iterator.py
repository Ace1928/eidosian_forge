import sys
import time
from typing import Optional
import numpy as np
import ray
from ray import train
from ray.air.config import DatasetConfig, ScalingConfig
from ray.data import Dataset, DataIterator, Preprocessor
from ray.train.data_parallel_trainer import DataParallelTrainer
from ray.train import DataConfig
from ray.util.annotations import Deprecated, DeveloperAPI
@Deprecated(MAKE_LOCAL_DATA_ITERATOR_DEPRECATION_MSG)
def make_local_dataset_iterator(dataset: Dataset, preprocessor: Preprocessor, dataset_config: DatasetConfig) -> DataIterator:
    """A helper function to create a local
    :py:class:`DataIterator <ray.data.DataIterator>`,
    like the one returned by :meth:`~ray.train.get_dataset_shard`.

    This function should only be used for development and debugging. It will
    raise an exception if called by a worker instead of the driver.

    Args:
        dataset: The input Dataset.
        preprocessor: The preprocessor that will be applied to the input dataset.
        dataset_config: The dataset config normally passed to the trainer.
    """
    raise DeprecationWarning(MAKE_LOCAL_DATA_ITERATOR_DEPRECATION_MSG)
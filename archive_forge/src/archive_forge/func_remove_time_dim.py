import numpy as np
import pandas as pd
from typing import Any, Dict, Type, TYPE_CHECKING
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.annotations import DeveloperAPI
@DeveloperAPI
def remove_time_dim(batch: pd.DataFrame) -> pd.DataFrame:
    """Removes the time dimension from the given sub-batch of the dataset.

    If each row in a dataset has a time dimension ([T, D]), and T=1, this function will
    remove the T dimension to convert each row to of shape [D]. If T > 1, the row is
    left unchanged. This function is to be used with map_batches().

    Args:
        batch: The batch to remove the time dimension from.
    Returns:
        The modified batch with the time dimension removed (when applicable)
    """
    BATCHED_KEYS = {SampleBatch.OBS, SampleBatch.ACTIONS, SampleBatch.ACTION_PROB, SampleBatch.REWARDS, SampleBatch.NEXT_OBS, SampleBatch.DONES}
    for k in batch.columns:
        if k in BATCHED_KEYS:
            batch[k] = batch[k].apply(lambda x: x[0] if len(x) == 1 else x)
    return batch
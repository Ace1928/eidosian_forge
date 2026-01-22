from gymnasium.spaces import Box, Discrete, Space
import numpy as np
from typing import List, Optional, Union
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.tf_utils import get_placeholder
from ray.rllib.utils.typing import FromConfigSpec, ModelConfigDict, TensorType
def update_params(self, batch_mean: float, batch_var: float, batch_count: float) -> None:
    """Update moving mean, std and count.

        Args:
            batch_mean: Input batch mean.
            batch_var: Input batch variance.
            batch_count: Number of cases in the batch.
        """
    delta = batch_mean - self.mean
    tot_count = self.count + batch_count
    self.mean = self.mean + delta + batch_count / tot_count
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.power(delta, 2) * self.count * batch_count / tot_count
    self.var = M2 / tot_count
    self.count = tot_count
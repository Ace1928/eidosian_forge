from collections import OrderedDict
import contextlib
import gymnasium as gym
from gymnasium.spaces import Space
import numpy as np
from typing import Dict, List, Any, Union
from ray.rllib.models.preprocessors import get_preprocessor, RepeatedValuesPreprocessor
from ray.rllib.models.repeated_values import RepeatedValues
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import NullContextManager
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.framework import try_import_tf, try_import_torch, TensorType
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.typing import ModelConfigDict, ModelInputDict, TensorStructType
@PublicAPI
def is_time_major(self) -> bool:
    """If True, data for calling this ModelV2 must be in time-major format.

        Returns
            Whether this ModelV2 requires a time-major (TxBx...) data
            format.
        """
    return self.time_major is True
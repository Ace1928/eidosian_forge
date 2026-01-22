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
@DeveloperAPI
def restore_original_dimensions(obs: TensorType, obs_space: Space, tensorlib: Any=tf) -> TensorStructType:
    """Unpacks Dict and Tuple space observations into their original form.

    This is needed since we flatten Dict and Tuple observations in transit
    within a SampleBatch. Before sending them to the model though, we should
    unflatten them into Dicts or Tuples of tensors.

    Args:
        obs: The flattened observation tensor.
        obs_space: The flattened obs space. If this has the
            `original_space` attribute, we will unflatten the tensor to that
            shape.
        tensorlib: The library used to unflatten (reshape) the array/tensor.

    Returns:
        single tensor or dict / tuple of tensors matching the original
        observation space.
    """
    if tensorlib in ['tf', 'tf2']:
        assert tf is not None
        tensorlib = tf
    elif tensorlib == 'torch':
        assert torch is not None
        tensorlib = torch
    elif tensorlib == 'numpy':
        assert np is not None
        tensorlib = np
    original_space = getattr(obs_space, 'original_space', obs_space)
    return _unpack_obs(obs, original_space, tensorlib=tensorlib)
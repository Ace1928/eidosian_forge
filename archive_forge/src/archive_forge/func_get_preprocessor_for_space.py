from functools import partial
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
import logging
import numpy as np
import tree  # pip install dm_tree
from typing import List, Optional, Type, Union
from ray.tune.registry import (
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor
from ray.rllib.models.tf.tf_action_dist import (
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.typing import ModelConfigDict, TensorType
@staticmethod
@DeveloperAPI
def get_preprocessor_for_space(observation_space: gym.Space, options: dict=None, include_multi_binary: bool=False) -> Preprocessor:
    """Returns a suitable preprocessor for the given observation space.

        Args:
            observation_space: The input observation space.
            options: Options to pass to the preprocessor.
            include_multi_binary: Whether to include the MultiBinaryPreprocessor in
                the possible preprocessors returned by this method.

        Returns:
            preprocessor: Preprocessor for the observations.
        """
    options = options or MODEL_DEFAULTS
    for k in options.keys():
        if k not in MODEL_DEFAULTS:
            raise Exception('Unknown config key `{}`, all keys: {}'.format(k, list(MODEL_DEFAULTS)))
    cls = get_preprocessor(observation_space, include_multi_binary=include_multi_binary)
    prep = cls(observation_space, options)
    if prep is not None:
        logger.debug('Created preprocessor {}: {} -> {}'.format(prep, observation_space, prep.shape))
    return prep
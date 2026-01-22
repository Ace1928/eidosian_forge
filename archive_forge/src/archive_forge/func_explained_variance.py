import logging
from typing import Any, Callable, List, Optional, Type, TYPE_CHECKING, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Discrete, MultiDiscrete
from ray.rllib.utils.annotations import PublicAPI, DeveloperAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import (
@PublicAPI
def explained_variance(y: TensorType, pred: TensorType) -> TensorType:
    """Computes the explained variance for a pair of labels and predictions.

    The formula used is:
    max(-1.0, 1.0 - (std(y - pred)^2 / std(y)^2))

    Args:
        y: The labels.
        pred: The predictions.

    Returns:
        The explained variance given a pair of labels and predictions.
    """
    _, y_var = tf.nn.moments(y, axes=[0])
    _, diff_var = tf.nn.moments(y - pred, axes=[0])
    return tf.maximum(-1.0, 1 - diff_var / y_var)
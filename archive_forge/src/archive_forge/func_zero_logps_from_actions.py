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
def zero_logps_from_actions(actions: TensorStructType) -> TensorType:
    """Helper function useful for returning dummy logp's (0) for some actions.

    Args:
        actions: The input actions. This can be any struct
            of complex action components or a simple tensor of different
            dimensions, e.g. [B], [B, 2], or {"a": [B, 4, 5], "b": [B]}.

    Returns:
        A 1D tensor of 0.0 (dummy logp's) matching the batch
        dim of `actions` (shape=[B]).
    """
    action_component = tree.flatten(actions)[0]
    logp_ = tf.zeros_like(action_component, dtype=tf.float32)
    while len(logp_.shape) > 1:
        logp_ = logp_[:, 0]
    return logp_
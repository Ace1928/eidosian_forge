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
def get_gpu_devices() -> List[str]:
    """Returns a list of GPU device names, e.g. ["/gpu:0", "/gpu:1"].

    Supports both tf1.x and tf2.x.

    Returns:
        List of GPU device names (str).
    """
    if tfv == 1:
        from tensorflow.python.client import device_lib
        devices = device_lib.list_local_devices()
    else:
        try:
            devices = tf.config.list_physical_devices()
        except Exception:
            devices = tf.config.experimental.list_physical_devices()
    return [d.name for d in devices if 'GPU' in d.device_type]
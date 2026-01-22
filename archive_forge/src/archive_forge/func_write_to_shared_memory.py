import multiprocessing as mp
from collections import OrderedDict
from ctypes import c_bool
from functools import singledispatch
from typing import Union
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@singledispatch
def write_to_shared_memory(space: Space, index: int, value: np.ndarray, shared_memory: Union[dict, tuple, mp.Array]):
    """Write the observation of a single environment into shared memory.

    Args:
        space: Observation space of a single environment in the vectorized environment.
        index: Index of the environment (must be in `[0, num_envs)`).
        value: Observation of the single environment to write to shared memory.
        shared_memory: Shared object across processes. This contains the observations from the vectorized environment.
            This object is created with `create_shared_memory`.

    Raises:
        CustomSpaceError: Space is not a valid :class:`gym.Space` instance
    """
    raise CustomSpaceError(f'Cannot write to a shared memory for space with type `{type(space)}`. Shared memory only supports default Gym spaces (e.g. `Box`, `Tuple`, `Dict`, etc...), and does not support custom Gym spaces.')
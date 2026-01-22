from collections import OrderedDict
from copy import deepcopy
from functools import singledispatch
from typing import Iterator
import numpy as np
from gym.error import CustomSpaceError
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
@singledispatch
def batch_space(space: Space, n: int=1) -> Space:
    """Create a (batched) space, containing multiple copies of a single space.

    Example::

        >>> from gym.spaces import Box, Dict
        >>> space = Dict({
        ...     'position': Box(low=0, high=1, shape=(3,), dtype=np.float32),
        ...     'velocity': Box(low=0, high=1, shape=(2,), dtype=np.float32)
        ... })
        >>> batch_space(space, n=5)
        Dict(position:Box(5, 3), velocity:Box(5, 2))

    Args:
        space: Space (e.g. the observation space) for a single environment in the vectorized environment.
        n: Number of environments in the vectorized environment.

    Returns:
        Space (e.g. the observation space) for a batch of environments in the vectorized environment.

    Raises:
        ValueError: Cannot batch space that is not a valid :class:`gym.Space` instance
    """
    raise ValueError(f'Cannot batch space with type `{type(space)}`. The space must be a valid `gym.Space` instance.')
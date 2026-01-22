import gymnasium as gym
from gymnasium.spaces import Tuple, Dict
import numpy as np
from ray.rllib.utils.annotations import DeveloperAPI
import tree  # pip install dm_tree
from typing import Any, List, Optional, Union
@DeveloperAPI
def get_dummy_batch_for_space(space: gym.Space, batch_size: int=32, fill_value: Union[float, int, str]=0.0, time_size: Optional[int]=None, time_major: bool=False, one_hot_discrete: bool=False) -> np.ndarray:
    """Returns batched dummy data (using `batch_size`) for the given `space`.

    Note: The returned batch will not pass a `space.contains(batch)` test
    as an additional batch dimension has to be added at axis 0, unless `batch_size` is
    set to 0.

    Args:
        space: The space to get a dummy batch for.
        batch_size: The required batch size (B). Note that this can also
            be 0 (only if `time_size` is None!), which will result in a
            non-batched sample for the given space (no batch dim).
        fill_value: The value to fill the batch with
            or "random" for random values.
        time_size: If not None, add an optional time axis
            of `time_size` size to the returned batch. This time axis might either
            be inserted at axis=1 (default) or axis=0, if `time_major` is True.
        time_major: If True AND `time_size` is not None, return batch
            as shape [T x B x ...], otherwise as [B x T x ...]. If `time_size`
            if None, ignore this setting and return [B x ...].
        one_hot_discrete: If True, will return one-hot vectors (instead of
            int-values) for those sub-components of a (possibly complex) `space`
            that are Discrete or MultiDiscrete. Note that in case `fill_value` is 0.0,
            this will result in zero-hot vectors (where all slots have a value of 0.0).

    Returns:
        The dummy batch of size `bqtch_size` matching the given space.
    """
    if isinstance(space, (gym.spaces.Dict, gym.spaces.Tuple, dict, tuple)):
        base_struct = space
        if isinstance(space, (gym.spaces.Dict, gym.spaces.Tuple)):
            base_struct = get_base_struct_from_space(space)
        return tree.map_structure(lambda s: get_dummy_batch_for_space(space=s, batch_size=batch_size, fill_value=fill_value, time_size=time_size, time_major=time_major, one_hot_discrete=one_hot_discrete), base_struct)
    if one_hot_discrete:
        if isinstance(space, gym.spaces.Discrete):
            space = gym.spaces.Box(0.0, 1.0, (space.n,), np.float32)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            space = gym.spaces.Box(0.0, 1.0, (np.sum(space.nvec),), np.float32)
    if fill_value == 'random':
        if time_size is not None:
            assert batch_size > 0 and time_size > 0
            if time_major:
                return np.array([[space.sample() for _ in range(batch_size)] for t in range(time_size)], dtype=space.dtype)
            else:
                return np.array([[space.sample() for t in range(time_size)] for _ in range(batch_size)], dtype=space.dtype)
        else:
            return np.array([space.sample() for _ in range(batch_size)] if batch_size > 0 else space.sample(), dtype=space.dtype)
    else:
        if time_size is not None:
            assert batch_size > 0 and time_size > 0
            if time_major:
                shape = [time_size, batch_size]
            else:
                shape = [batch_size, time_size]
        else:
            shape = [batch_size] if batch_size > 0 else []
        return np.full(shape + list(space.shape), fill_value=fill_value, dtype=space.dtype)
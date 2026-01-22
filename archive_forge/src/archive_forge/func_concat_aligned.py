from collections import OrderedDict
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
import tree  # pip install dm_tree
from types import MappingProxyType
from typing import List, Optional
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import SpaceStruct, TensorType, TensorStructType, Union
@PublicAPI
@Deprecated(help='RLlib itself has no use for this anymore.', error=False)
def concat_aligned(items: List[np.ndarray], time_major: Optional[bool]=None) -> np.ndarray:
    """Concatenate arrays, ensuring the output is 64-byte aligned.

    We only align float arrays; other arrays are concatenated as normal.

    This should be used instead of np.concatenate() to improve performance
    when the output array is likely to be fed into TensorFlow.

    Args:
        items: The list of items to concatenate and align.
        time_major: Whether the data in items is time-major, in which
            case, we will concatenate along axis=1.

    Returns:
        The concat'd and aligned array.
    """
    if len(items) == 0:
        return []
    elif len(items) == 1:
        return items[0]
    elif isinstance(items[0], np.ndarray) and items[0].dtype in [np.float32, np.float64, np.uint8]:
        dtype = items[0].dtype
        flat = aligned_array(sum((s.size for s in items)), dtype)
        if time_major is not None:
            if time_major is True:
                batch_dim = sum((s.shape[1] for s in items))
                new_shape = (items[0].shape[0], batch_dim) + items[0].shape[2:]
            else:
                batch_dim = sum((s.shape[0] for s in items))
                new_shape = (batch_dim, items[0].shape[1]) + items[0].shape[2:]
        else:
            batch_dim = sum((s.shape[0] for s in items))
            new_shape = (batch_dim,) + items[0].shape[1:]
        output = flat.reshape(new_shape)
        assert output.ctypes.data % 64 == 0, output.ctypes.data
        np.concatenate(items, out=output, axis=1 if time_major else 0)
        return output
    else:
        return np.concatenate(items, axis=1 if time_major else 0)
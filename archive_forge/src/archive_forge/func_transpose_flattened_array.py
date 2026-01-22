import dataclasses
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from cirq import protocols
from cirq.linalg import predicates
def transpose_flattened_array(t: np.ndarray, shape: Sequence[int], axes: Sequence[int]):
    """Transposes a flattened array.

    Equivalent to np.transpose(t.reshape(shape), axes).reshape((-1,)).

    Args:
        t: flat array.
        shape: the shape of `t` before flattening.
        axes: permutation of range(len(shape)).

    Returns:
        Flattened transpose of `t`.
    """
    if len(t.shape) != 1:
        t = t.reshape((-1,))
    cur_volume = _volumes(shape)
    new_volume = _volumes([shape[i] for i in axes])
    ret = np.zeros_like(t)
    for idx in range(t.shape[0]):
        cell = _coordinates_from_index(idx, cur_volume)
        new_cell = [cell[i] for i in axes]
        ret[_index_from_coordinates(new_cell, new_volume)] = t[idx]
    return ret
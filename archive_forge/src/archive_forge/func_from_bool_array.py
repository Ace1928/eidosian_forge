from __future__ import annotations
from collections import defaultdict
from functools import partial
from itertools import chain, repeat
from typing import Callable, Iterable, Literal, Mapping
import numpy as np
from numpy.typing import NDArray
from qiskit.result import Counts
from .shape import ShapedMixin, ShapeInput, shape_tuple
@staticmethod
def from_bool_array(array: NDArray[np.bool_], order: Literal['big', 'little']='big') -> 'BitArray':
    """Construct a new bit array from an array of bools.

        Args:
            array: The array to convert, with "bitstrings" along the last axis.
            order: One of ``"big"`` or ``"little"``, indicating whether ``array[..., 0]``
                correspond to the most significant bits or the least significant bits of each
                bitstring, respectively.

        Returns:
            A new bit array.
        """
    array = np.asarray(array, dtype=bool)
    if array.ndim < 2:
        raise ValueError('Expecting at least two dimensions.')
    if order == 'little':
        array = array[..., ::-1]
    num_bits = array.shape[-1]
    if (remainder := (-num_bits % 8)):
        pad = np.zeros(shape_tuple(array.shape[:-1], remainder), dtype=bool)
        array = np.concatenate([pad, array], axis=-1)
    return BitArray(np.packbits(array, axis=-1), num_bits=num_bits)
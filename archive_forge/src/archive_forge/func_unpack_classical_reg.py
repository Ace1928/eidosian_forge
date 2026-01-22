from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
def unpack_classical_reg(c: MemoryReferenceDesignator) -> 'MemoryReference':
    """
    Get the address for a classical register.

    :param c: A list of length 2, a pair, a string (to be interpreted as name[0]), or a
        MemoryReference.
    :return: The address as a MemoryReference.
    """
    if isinstance(c, list) or isinstance(c, tuple):
        if len(c) > 2 or len(c) == 0:
            raise ValueError('if c is a list/tuple, it should be of length <= 2')
        if len(c) == 1:
            c = (c[0], 0)
        if not isinstance(c[0], str):
            raise ValueError('if c is a list/tuple, its first member should be a string')
        if not isinstance(c[1], int):
            raise ValueError('if c is a list/tuple, its second member should be an int')
        return MemoryReference(c[0], c[1])
    if isinstance(c, MemoryReference):
        return c
    elif isinstance(c, str):
        return MemoryReference(c, 0)
    else:
        raise TypeError('c should be a list of length 2, a pair, a string, or a MemoryReference')
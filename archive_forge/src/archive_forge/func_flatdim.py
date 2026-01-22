import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@singledispatch
def flatdim(space: Space) -> int:
    """Return the number of dimensions a flattened equivalent of this space would have.

    Example usage::

        >>> from gym.spaces import Discrete
        >>> space = Dict({"position": Discrete(2), "velocity": Discrete(3)})
        >>> flatdim(space)
        5

    Args:
        space: The space to return the number of dimensions of the flattened spaces

    Returns:
        The number of dimensions for the flattened spaces

    Raises:
         NotImplementedError: if the space is not defined in ``gym.spaces``.
         ValueError: if the space cannot be flattened into a :class:`Box`
    """
    if not space.is_np_flattenable:
        raise ValueError(f'{space} cannot be flattened to a numpy array, probably because it contains a `Graph` or `Sequence` subspace')
    raise NotImplementedError(f'Unknown space: `{space}`')
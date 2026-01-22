import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
def normalize_1d_index(index):
    """
    Normalize the *index* type (an integer or slice) for indexing a 1D
    sequence.
    """
    if isinstance(index, types.SliceType):
        return index
    elif isinstance(index, types.Integer):
        return types.intp if index.signed else types.uintp
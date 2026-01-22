import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
def choose_result_int(*inputs):
    """
    Choose the integer result type for an operation on integer inputs,
    according to the integer typing NBEP.
    """
    bitwidth = choose_result_bitwidth(*inputs)
    signed = any((tp.signed for tp in inputs))
    return types.Integer.from_bitwidth(bitwidth, signed)
from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def is_fprm_sort(s):
    """Return True if `s` is a Z3 floating-point rounding mode sort.

    >>> is_fprm_sort(FPSort(8, 24))
    False
    >>> is_fprm_sort(RNE().sort())
    True
    """
    return isinstance(s, FPRMSortRef)
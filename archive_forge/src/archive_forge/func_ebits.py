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
def ebits(self):
    """Retrieves the number of bits reserved for the exponent in the FloatingPoint expression `self`.
        >>> b = FPSort(8, 24)
        >>> b.ebits()
        8
        """
    return self.sort().ebits()
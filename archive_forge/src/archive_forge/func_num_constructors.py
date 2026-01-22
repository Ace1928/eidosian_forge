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
def num_constructors(self):
    """Return the number of constructors in the given Z3 datatype.

        >>> List = Datatype('List')
        >>> List.declare('cons', ('car', IntSort()), ('cdr', List))
        >>> List.declare('nil')
        >>> List = List.create()
        >>> # List is now a Z3 declaration
        >>> List.num_constructors()
        2
        """
    return int(Z3_get_datatype_sort_num_constructors(self.ctx_ref(), self.ast))
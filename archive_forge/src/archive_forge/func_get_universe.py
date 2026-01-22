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
def get_universe(self, s):
    """Return the interpretation for the uninterpreted sort `s` in the model `self`.

        >>> A = DeclareSort('A')
        >>> a, b = Consts('a b', A)
        >>> s = Solver()
        >>> s.add(a != b)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m.get_universe(A)
        [A!val!1, A!val!0]
        """
    if z3_debug():
        _z3_assert(isinstance(s, SortRef), 'Z3 sort expected')
    try:
        return AstVector(Z3_model_get_sort_universe(self.ctx.ref(), self.model, s.ast), self.ctx)
    except Z3Exception:
        return None
from __future__ import annotations
from typing import Any
from functools import reduce
from itertools import permutations
from sympy.combinatorics import Permutation
from sympy.core import (
from sympy.core.cache import cacheit
from sympy.core.symbol import Symbol, Dummy
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.functions import factorial
from sympy.matrices import ImmutableDenseMatrix as Matrix
from sympy.solvers import solve
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.simplify.simplify import simplify
def coord_tuple_transform_to(self, to_sys, coords):
    """Transform ``coords`` to coord system ``to_sys``."""
    sympy_deprecation_warning('\n            The CoordSystem.coord_tuple_transform_to() method is deprecated.\n            Use the CoordSystem.transform() method instead.\n            ', deprecated_since_version='1.7', active_deprecations_target='deprecated-diffgeom-mutable')
    coords = Matrix(coords)
    if self != to_sys:
        with ignore_warnings(SymPyDeprecationWarning):
            transf = self.transforms[to_sys]
        coords = transf[1].subs(list(zip(transf[0], coords)))
    return coords
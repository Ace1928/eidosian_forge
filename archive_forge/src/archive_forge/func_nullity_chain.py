from types import FunctionType
from collections import Counter
from mpmath import mp, workprec
from mpmath.libmp.libmpf import prec_to_dps
from sympy.core.sorting import default_sort_key
from sympy.core.evalf import DEFAULT_MAXPREC, PrecisionExhausted
from sympy.core.logic import fuzzy_and, fuzzy_or
from sympy.core.numbers import Float
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys import roots, CRootOf, ZZ, QQ, EX
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.eigen import dom_eigenvects, dom_eigenvects_to_sympy
from sympy.polys.polytools import gcd
from .common import MatrixError, NonSquareMatrixError
from .determinant import _find_reasonable_pivot
from .utilities import _iszero, _simplify
def nullity_chain(val, algebraic_multiplicity):
    """Calculate the sequence  [0, nullity(E), nullity(E**2), ...]
        until it is constant where ``E = M - val*I``"""
    cols = M.cols
    ret = [0]
    nullity = cols - eig_mat(val, 1).rank()
    i = 2
    while nullity != ret[-1]:
        ret.append(nullity)
        if nullity == algebraic_multiplicity:
            break
        nullity = cols - eig_mat(val, i).rank()
        i += 1
        if nullity < ret[-1] or nullity > algebraic_multiplicity:
            raise MatrixError('SymPy had encountered an inconsistent result while computing Jordan block: {}'.format(M))
    return ret
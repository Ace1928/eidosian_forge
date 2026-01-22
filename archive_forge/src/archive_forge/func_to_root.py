from __future__ import annotations
import numbers
import decimal
import fractions
import math
import re as regex
import sys
from functools import lru_cache
from .containers import Tuple
from .sympify import (SympifyError, _sympy_converter, sympify, _convert_numpy_types,
from .singleton import S, Singleton
from .basic import Basic
from .expr import Expr, AtomicExpr
from .evalf import pure_complex
from .cache import cacheit, clear_cache
from .decorators import _sympifyit
from .logic import fuzzy_not
from .kind import NumberKind
from sympy.external.gmpy import SYMPY_INTS, HAS_GMPY, gmpy
from sympy.multipledispatch import dispatch
import mpmath
import mpmath.libmp as mlib
from mpmath.libmp import bitcount, round_nearest as rnd
from mpmath.libmp.backend import MPZ
from mpmath.libmp import mpf_pow, mpf_pi, mpf_e, phi_fixed
from mpmath.ctx_mp import mpnumeric
from mpmath.libmp.libmpf import (
from sympy.utilities.misc import as_int, debug, filldedent
from .parameters import global_parameters
from .power import Pow, integer_nthroot
from .mul import Mul
from .add import Add
def to_root(self, radicals=True, minpoly=None):
    """
        Convert to an :py:class:`~.Expr` that is not an
        :py:class:`~.AlgebraicNumber`, specifically, either a
        :py:class:`~.ComplexRootOf`, or, optionally and where possible, an
        expression in radicals.

        Parameters
        ==========

        radicals : boolean, optional (default=True)
            If ``True``, then we will try to return the root as an expression
            in radicals. If that is not possible, we will return a
            :py:class:`~.ComplexRootOf`.

        minpoly : :py:class:`~.Poly`
            If the minimal polynomial for `self` has been pre-computed, it can
            be passed in order to save time.

        """
    if self.is_primitive_element and (not isinstance(self.root, AlgebraicNumber)):
        return self.root
    m = minpoly or self.minpoly_of_element()
    roots = m.all_roots(radicals=radicals)
    if len(roots) == 1:
        return roots[0]
    ex = self.as_expr()
    for b in roots:
        if m.same_root(b, ex):
            return b
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
def minpoly_of_element(self):
    """
        Compute the minimal polynomial for this algebraic number.

        Explanation
        ===========

        Recall that we represent an element $\\alpha \\in \\mathbb{Q}(\\theta)$.
        Our instance attribute ``self.minpoly`` is the minimal polynomial for
        our primitive element $\\theta$. This method computes the minimal
        polynomial for $\\alpha$.

        """
    if self._own_minpoly is None:
        if self.is_primitive_element:
            self._own_minpoly = self.minpoly
        else:
            from sympy.polys.numberfields.minpoly import minpoly
            theta = self.primitive_element()
            self._own_minpoly = minpoly(self.as_expr(theta), polys=True)
    return self._own_minpoly
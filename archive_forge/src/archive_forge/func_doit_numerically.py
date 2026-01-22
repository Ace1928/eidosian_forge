from __future__ import annotations
from typing import Any
from collections.abc import Iterable
from .add import Add
from .basic import Basic, _atomic
from .cache import cacheit
from .containers import Tuple, Dict
from .decorators import _sympifyit
from .evalf import pure_complex
from .expr import Expr, AtomicExpr
from .logic import fuzzy_and, fuzzy_or, fuzzy_not, FuzzyBool
from .mul import Mul
from .numbers import Rational, Float, Integer
from .operations import LatticeOp
from .parameters import global_parameters
from .rules import Transform
from .singleton import S
from .sympify import sympify, _sympify
from .sorting import default_sort_key, ordered
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.utilities.iterables import (has_dups, sift, iterable,
from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
from sympy.utilities.misc import as_int, filldedent, func_name
import mpmath
from mpmath.libmp.libmpf import prec_to_dps
import inspect
from collections import Counter
from .symbol import Dummy, Symbol
@_sympifyit('z0', NotImplementedError)
def doit_numerically(self, z0):
    """
        Evaluate the derivative at z numerically.

        When we can represent derivatives at a point, this should be folded
        into the normal evalf. For now, we need a special method.
        """
    if len(self.free_symbols) != 1 or len(self.variables) != 1:
        raise NotImplementedError('partials and higher order derivatives')
    z = list(self.free_symbols)[0]

    def eval(x):
        f0 = self.expr.subs(z, Expr._from_mpmath(x, prec=mpmath.mp.prec))
        f0 = f0.evalf(prec_to_dps(mpmath.mp.prec))
        return f0._to_mpmath(mpmath.mp.prec)
    return Expr._from_mpmath(mpmath.diff(eval, z0._to_mpmath(mpmath.mp.prec)), mpmath.mp.prec)
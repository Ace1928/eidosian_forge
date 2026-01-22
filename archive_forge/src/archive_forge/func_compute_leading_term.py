from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Iterable
from functools import reduce
import re
from .sympify import sympify, _sympify
from .basic import Basic, Atom
from .singleton import S
from .evalf import EvalfMixin, pure_complex, DEFAULT_MAXPREC
from .decorators import call_highest_priority, sympify_method_args, sympify_return
from .cache import cacheit
from .sorting import default_sort_key
from .kind import NumberKind
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.misc import as_int, func_name, filldedent
from sympy.utilities.iterables import has_variety, sift
from mpmath.libmp import mpf_log, prec_to_dps
from mpmath.libmp.libintmath import giant_steps
from collections import defaultdict
from .mul import Mul
from .add import Add
from .power import Pow
from .function import Function, _derivative_dispatch
from .mod import Mod
from .exprtools import factor_terms
from .numbers import Float, Integer, Rational, _illegal
def compute_leading_term(self, x, logx=None):
    """Deprecated function to compute the leading term of a series.

        as_leading_term is only allowed for results of .series()
        This is a wrapper to compute a series first.
        """
    from sympy.utilities.exceptions import SymPyDeprecationWarning
    SymPyDeprecationWarning(feature='compute_leading_term', useinstead='as_leading_term', issue=21843, deprecated_since_version='1.12').warn()
    from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
    if self.has(Piecewise):
        expr = piecewise_fold(self)
    else:
        expr = self
    if self.removeO() == 0:
        return self
    from .symbol import Dummy
    from sympy.functions.elementary.exponential import log
    from sympy.series.order import Order
    _logx = logx
    logx = Dummy('logx') if logx is None else logx
    res = Order(1)
    incr = S.One
    while res.is_Order:
        res = expr._eval_nseries(x, n=1 + incr, logx=logx).cancel().powsimp().trigsimp()
        incr *= 2
    if _logx is None:
        res = res.subs(logx, log(x))
    return res.as_leading_term(x)
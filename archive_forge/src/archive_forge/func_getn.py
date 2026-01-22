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
def getn(self):
    """
        Returns the order of the expression.

        Explanation
        ===========

        The order is determined either from the O(...) term. If there
        is no O(...) term, it returns None.

        Examples
        ========

        >>> from sympy import O
        >>> from sympy.abc import x
        >>> (1 + x + O(x**2)).getn()
        2
        >>> (1 + x).getn()

        """
    o = self.getO()
    if o is None:
        return None
    elif o.is_Order:
        o = o.expr
        if o is S.One:
            return S.Zero
        if o.is_Symbol:
            return S.One
        if o.is_Pow:
            return o.args[1]
        if o.is_Mul:
            for oi in o.args:
                if oi.is_Symbol:
                    return S.One
                if oi.is_Pow:
                    from .symbol import Dummy, Symbol
                    syms = oi.atoms(Symbol)
                    if len(syms) == 1:
                        x = syms.pop()
                        oi = oi.subs(x, Dummy('x', positive=True))
                        if oi.base.is_Symbol and oi.exp.is_Rational:
                            return abs(oi.exp)
    raise NotImplementedError('not sure of order of %s' % o)
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
def extract_multiplicatively(self, c):
    """Return None if it's not possible to make self in the form
           c * something in a nice way, i.e. preserving the properties
           of arguments of self.

           Examples
           ========

           >>> from sympy import symbols, Rational

           >>> x, y = symbols('x,y', real=True)

           >>> ((x*y)**3).extract_multiplicatively(x**2 * y)
           x*y**2

           >>> ((x*y)**3).extract_multiplicatively(x**4 * y)

           >>> (2*x).extract_multiplicatively(2)
           x

           >>> (2*x).extract_multiplicatively(3)

           >>> (Rational(1, 2)*x).extract_multiplicatively(3)
           x/6

        """
    from sympy.functions.elementary.exponential import exp
    from .add import _unevaluated_Add
    c = sympify(c)
    if self is S.NaN:
        return None
    if c is S.One:
        return self
    elif c == self:
        return S.One
    if c.is_Add:
        cc, pc = c.primitive()
        if cc is not S.One:
            c = Mul(cc, pc, evaluate=False)
    if c.is_Mul:
        a, b = c.as_two_terms()
        x = self.extract_multiplicatively(a)
        if x is not None:
            return x.extract_multiplicatively(b)
        else:
            return x
    quotient = self / c
    if self.is_Number:
        if self is S.Infinity:
            if c.is_positive:
                return S.Infinity
        elif self is S.NegativeInfinity:
            if c.is_negative:
                return S.Infinity
            elif c.is_positive:
                return S.NegativeInfinity
        elif self is S.ComplexInfinity:
            if not c.is_zero:
                return S.ComplexInfinity
        elif self.is_Integer:
            if not quotient.is_Integer:
                return None
            elif self.is_positive and quotient.is_negative:
                return None
            else:
                return quotient
        elif self.is_Rational:
            if not quotient.is_Rational:
                return None
            elif self.is_positive and quotient.is_negative:
                return None
            else:
                return quotient
        elif self.is_Float:
            if not quotient.is_Float:
                return None
            elif self.is_positive and quotient.is_negative:
                return None
            else:
                return quotient
    elif self.is_NumberSymbol or self.is_Symbol or self is S.ImaginaryUnit:
        if quotient.is_Mul and len(quotient.args) == 2:
            if quotient.args[0].is_Integer and quotient.args[0].is_positive and (quotient.args[1] == self):
                return quotient
        elif quotient.is_Integer and c.is_Number:
            return quotient
    elif self.is_Add:
        cs, ps = self.primitive()
        if c.is_Number and c is not S.NegativeOne:
            if cs is not S.One:
                if c.is_negative:
                    xc = -cs.extract_multiplicatively(-c)
                else:
                    xc = cs.extract_multiplicatively(c)
                if xc is not None:
                    return xc * ps
            return
        if c == ps:
            return cs
        newargs = []
        for arg in ps.args:
            newarg = arg.extract_multiplicatively(c)
            if newarg is None:
                return
            newargs.append(newarg)
        if cs is not S.One:
            args = [cs * t for t in newargs]
            return _unevaluated_Add(*args)
        else:
            return Add._from_args(newargs)
    elif self.is_Mul:
        args = list(self.args)
        for i, arg in enumerate(args):
            newarg = arg.extract_multiplicatively(c)
            if newarg is not None:
                args[i] = newarg
                return Mul(*args)
    elif self.is_Pow or isinstance(self, exp):
        sb, se = self.as_base_exp()
        cb, ce = c.as_base_exp()
        if cb == sb:
            new_exp = se.extract_additively(ce)
            if new_exp is not None:
                return Pow(sb, new_exp)
        elif c == sb:
            new_exp = self.exp.extract_additively(1)
            if new_exp is not None:
                return Pow(sb, new_exp)
from functools import reduce
from sympy.core import S, ilcm, Mod
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Function, Derivative, ArgumentIndexError
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.numbers import I, pi, oo, zoo
from sympy.core.relational import Ne
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.functions import (sqrt, exp, log, sin, cos, asin, atan,
from sympy.functions import factorial, RisingFactorial
from sympy.functions.elementary.complexes import Abs, re, unpolarify
from sympy.functions.elementary.exponential import exp_polar
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import (And, Or)
@property
def radius_of_convergence(self):
    """
        Compute the radius of convergence of the defining series.

        Explanation
        ===========

        Note that even if this is not ``oo``, the function may still be
        evaluated outside of the radius of convergence by analytic
        continuation. But if this is zero, then the function is not actually
        defined anywhere else.

        Examples
        ========

        >>> from sympy import hyper
        >>> from sympy.abc import z
        >>> hyper((1, 2), [3], z).radius_of_convergence
        1
        >>> hyper((1, 2, 3), [4], z).radius_of_convergence
        0
        >>> hyper((1, 2), (3, 4), z).radius_of_convergence
        oo

        """
    if any((a.is_integer and (a <= 0) == True for a in self.ap + self.bq)):
        aints = [a for a in self.ap if a.is_Integer and (a <= 0) == True]
        bints = [a for a in self.bq if a.is_Integer and (a <= 0) == True]
        if len(aints) < len(bints):
            return S.Zero
        popped = False
        for b in bints:
            cancelled = False
            while aints:
                a = aints.pop()
                if a >= b:
                    cancelled = True
                    break
                popped = True
            if not cancelled:
                return S.Zero
        if aints or popped:
            return oo
    if len(self.ap) == len(self.bq) + 1:
        return S.One
    elif len(self.ap) <= len(self.bq):
        return oo
    else:
        return S.Zero
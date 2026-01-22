from typing import Tuple as tTuple
from sympy.concrete.expr_with_limits import AddWithLimits
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import diff
from sympy.core.logic import fuzzy_bool
from sympy.core.mul import Mul
from sympy.core.numbers import oo, pi
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild)
from sympy.core.sympify import sympify
from sympy.functions import Piecewise, sqrt, piecewise_fold, tan, cot, atan
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.miscellaneous import Min, Max
from .rationaltools import ratint
from sympy.matrices import MatrixBase
from sympy.polys import Poly, PolynomialError
from sympy.series.formal import FormalPowerSeries
from sympy.series.limits import limit
from sympy.series.order import Order
from sympy.tensor.functions import shape
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import filldedent
from .deltafunctions import deltaintegrate
from .meijerint import meijerint_definite, meijerint_indefinite, _debug
from .trigonometry import trigintegrate
def principal_value(self, **kwargs):
    """
        Compute the Cauchy Principal Value of the definite integral of a real function in the given interval
        on the real axis.

        Explanation
        ===========

        In mathematics, the Cauchy principal value, is a method for assigning values to certain improper
        integrals which would otherwise be undefined.

        Examples
        ========

        >>> from sympy import Integral, oo
        >>> from sympy.abc import x
        >>> Integral(x+1, (x, -oo, oo)).principal_value()
        oo
        >>> f = 1 / (x**3)
        >>> Integral(f, (x, -oo, oo)).principal_value()
        0
        >>> Integral(f, (x, -10, 10)).principal_value()
        0
        >>> Integral(f, (x, -10, oo)).principal_value() + Integral(f, (x, -oo, 10)).principal_value()
        0

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Cauchy_principal_value
        .. [2] https://mathworld.wolfram.com/CauchyPrincipalValue.html
        """
    if len(self.limits) != 1 or len(list(self.limits[0])) != 3:
        raise ValueError("You need to insert a variable, lower_limit, and upper_limit correctly to calculate cauchy's principal value")
    x, a, b = self.limits[0]
    if not (a.is_comparable and b.is_comparable and (a <= b)):
        raise ValueError("The lower_limit must be smaller than or equal to the upper_limit to calculate cauchy's principal value. Also, a and b need to be comparable.")
    if a == b:
        return S.Zero
    from sympy.calculus.singularities import singularities
    r = Dummy('r')
    f = self.function
    singularities_list = [s for s in singularities(f, x) if s.is_comparable and a <= s <= b]
    for i in singularities_list:
        if i in (a, b):
            raise ValueError('The principal value is not defined in the given interval due to singularity at %d.' % i)
    F = integrate(f, x, **kwargs)
    if F.has(Integral):
        return self
    if a is -oo and b is oo:
        I = limit(F - F.subs(x, -x), x, oo)
    else:
        I = limit(F, x, b, '-') - limit(F, x, a, '+')
    for s in singularities_list:
        I += limit(F.subs(x, s - r) - F.subs(x, s + r), r, 0, '+')
    return I
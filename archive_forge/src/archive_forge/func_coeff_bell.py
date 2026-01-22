from collections import defaultdict
from sympy.core.numbers import (nan, oo, zoo)
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Derivative, Function, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.sets.sets import Interval
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy, symbols, Symbol
from sympy.core.sympify import sympify
from sympy.discrete.convolutions import convolution
from sympy.functions.combinatorial.factorials import binomial, factorial, rf
from sympy.functions.combinatorial.numbers import bell
from sympy.functions.elementary.integers import floor, frac, ceiling
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.series.limits import Limit
from sympy.series.order import Order
from sympy.series.sequences import sequence
from sympy.series.series_class import SeriesBase
from sympy.utilities.iterables import iterable
def coeff_bell(self, n):
    """
        self.coeff_bell(n) returns a sequence of Bell polynomials of the second kind.
        Note that ``n`` should be a integer.

        The second kind of Bell polynomials (are sometimes called "partial" Bell
        polynomials or incomplete Bell polynomials) are defined as

        .. math::
            B_{n,k}(x_1, x_2,\\dotsc x_{n-k+1}) =
                \\sum_{j_1+j_2+j_2+\\dotsb=k \\atop j_1+2j_2+3j_2+\\dotsb=n}
                \\frac{n!}{j_1!j_2!\\dotsb j_{n-k+1}!}
                \\left(\\frac{x_1}{1!} \\right)^{j_1}
                \\left(\\frac{x_2}{2!} \\right)^{j_2} \\dotsb
                \\left(\\frac{x_{n-k+1}}{(n-k+1)!} \\right) ^{j_{n-k+1}}.

        * ``bell(n, k, (x1, x2, ...))`` gives Bell polynomials of the second kind,
          `B_{n,k}(x_1, x_2, \\dotsc, x_{n-k+1})`.

        See Also
        ========

        sympy.functions.combinatorial.numbers.bell

        """
    inner_coeffs = [bell(n, j, tuple(self.bell_coeff_seq[:n - j + 1])) for j in range(1, n + 1)]
    k = Dummy('k')
    return sequence(tuple(inner_coeffs), (k, 1, oo))
from sympy.core.numbers import (oo, pi)
from sympy.core.symbol import Wild
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import sin, cos, sinc
from sympy.series.series_class import SeriesBase
from sympy.series.sequences import SeqFormula
from sympy.sets.sets import Interval
from sympy.utilities.iterables import is_sequence
def sigma_approximation(self, n=3):
    """
        Return :math:`\\sigma`-approximation of Fourier series with respect
        to order n.

        Explanation
        ===========

        Sigma approximation adjusts a Fourier summation to eliminate the Gibbs
        phenomenon which would otherwise occur at discontinuities.
        A sigma-approximated summation for a Fourier series of a T-periodical
        function can be written as

        .. math::
            s(\\theta) = \\frac{1}{2} a_0 + \\sum _{k=1}^{m-1}
            \\operatorname{sinc} \\Bigl( \\frac{k}{m} \\Bigr) \\cdot
            \\left[ a_k \\cos \\Bigl( \\frac{2\\pi k}{T} \\theta \\Bigr)
            + b_k \\sin \\Bigl( \\frac{2\\pi k}{T} \\theta \\Bigr) \\right],

        where :math:`a_0, a_k, b_k, k=1,\\ldots,{m-1}` are standard Fourier
        series coefficients and
        :math:`\\operatorname{sinc} \\Bigl( \\frac{k}{m} \\Bigr)` is a Lanczos
        :math:`\\sigma` factor (expressed in terms of normalized
        :math:`\\operatorname{sinc}` function).

        Parameters
        ==========

        n : int
            Highest order of the terms taken into account in approximation.

        Returns
        =======

        Expr :
            Sigma approximation of function expanded into Fourier series.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x, (x, -pi, pi))
        >>> s.sigma_approximation(4)
        2*sin(x)*sinc(pi/4) - 2*sin(2*x)/pi + 2*sin(3*x)*sinc(3*pi/4)/3

        See Also
        ========

        sympy.series.fourier.FourierSeries.truncate

        Notes
        =====

        The behaviour of
        :meth:`~sympy.series.fourier.FourierSeries.sigma_approximation`
        is different from :meth:`~sympy.series.fourier.FourierSeries.truncate`
        - it takes all nonzero terms of degree smaller than n, rather than
        first n nonzero ones.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Gibbs_phenomenon
        .. [2] https://en.wikipedia.org/wiki/Sigma_approximation
        """
    terms = [sinc(pi * i / n) * t for i, t in enumerate(self[:n]) if t is not S.Zero]
    return Add(*terms)
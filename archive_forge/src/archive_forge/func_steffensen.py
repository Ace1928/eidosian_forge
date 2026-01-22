from __future__ import print_function
from copy import copy
from ..libmp.backend import xrange
def steffensen(f):
    """
    linear convergent function -> quadratic convergent function

    Steffensen's method for quadratic convergence of a linear converging
    sequence.
    Don not use it for higher rates of convergence.
    It may even work for divergent sequences.

    Definition:
    F(x) = (x*f(f(x)) - f(x)**2) / (f(f(x)) - 2*f(x) + x)

    Example
    .......

    You can use Steffensen's method to accelerate a fixpoint iteration of linear
    (or less) convergence.

    x* is a fixpoint of the iteration x_{k+1} = phi(x_k) if x* = phi(x*). For
    phi(x) = x**2 there are two fixpoints: 0 and 1.

    Let's try Steffensen's method:

    >>> f = lambda x: x**2
    >>> from mpmath.calculus.optimization import steffensen
    >>> F = steffensen(f)
    >>> for x in [0.5, 0.9, 2.0]:
    ...     fx = Fx = x
    ...     for i in xrange(9):
    ...         try:
    ...             fx = f(fx)
    ...         except OverflowError:
    ...             pass
    ...         try:
    ...             Fx = F(Fx)
    ...         except ZeroDivisionError:
    ...             pass
    ...         print('%20g  %20g' % (fx, Fx))
                    0.25                  -0.5
                  0.0625                   0.1
              0.00390625            -0.0011236
             1.52588e-05           1.41691e-09
             2.32831e-10          -2.84465e-27
             5.42101e-20           2.30189e-80
             2.93874e-39          -1.2197e-239
             8.63617e-78                     0
            7.45834e-155                     0
                    0.81               1.02676
                  0.6561               1.00134
                0.430467                     1
                0.185302                     1
               0.0343368                     1
              0.00117902                     1
             1.39008e-06                     1
             1.93233e-12                     1
             3.73392e-24                     1
                       4                   1.6
                      16                1.2962
                     256               1.10194
                   65536               1.01659
             4.29497e+09               1.00053
             1.84467e+19                     1
             3.40282e+38                     1
             1.15792e+77                     1
            1.34078e+154                     1

    Unmodified, the iteration converges only towards 0. Modified it converges
    not only much faster, it converges even to the repelling fixpoint 1.
    """

    def F(x):
        fx = f(x)
        ffx = f(fx)
        return (x * ffx - fx ** 2) / (ffx - 2 * fx + x)
    return F
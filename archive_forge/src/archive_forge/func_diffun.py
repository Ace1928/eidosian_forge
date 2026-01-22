from ..libmp.backend import xrange
from .calculus import defun
@defun
def diffun(ctx, f, n=1, **options):
    """
    Given a function `f`, returns a function `g(x)` that evaluates the nth
    derivative `f^{(n)}(x)`::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> cos2 = diffun(sin)
        >>> sin2 = diffun(sin, 4)
        >>> cos(1.3), cos2(1.3)
        (0.267498828624587, 0.267498828624587)
        >>> sin(1.3), sin2(1.3)
        (0.963558185417193, 0.963558185417193)

    The function `f` must support arbitrary precision evaluation.
    See :func:`~mpmath.diff` for additional details and supported
    keyword options.
    """
    if n == 0:
        return f

    def g(x):
        return ctx.diff(f, x, n, **options)
    return g
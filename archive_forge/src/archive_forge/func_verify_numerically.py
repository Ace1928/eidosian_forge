import ``rng`` and access the method directly. For example, to capture the
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int
import random as _random
def verify_numerically(f, g, z=None, tol=1e-06, a=2, b=-1, c=3, d=1):
    """
    Test numerically that f and g agree when evaluated in the argument z.

    If z is None, all symbols will be tested. This routine does not test
    whether there are Floats present with precision higher than 15 digits
    so if there are, your results may not be what you expect due to round-
    off errors.

    Examples
    ========

    >>> from sympy import sin, cos
    >>> from sympy.abc import x
    >>> from sympy.core.random import verify_numerically as tn
    >>> tn(sin(x)**2 + cos(x)**2, 1, x)
    True
    """
    from sympy.core.symbol import Symbol
    from sympy.core.sympify import sympify
    from sympy.core.numbers import comp
    f, g = (sympify(i) for i in (f, g))
    if z is None:
        z = f.free_symbols | g.free_symbols
    elif isinstance(z, Symbol):
        z = [z]
    reps = list(zip(z, [random_complex_number(a, b, c, d) for _ in z]))
    z1 = f.subs(reps).n()
    z2 = g.subs(reps).n()
    return comp(z1, z2, tol)
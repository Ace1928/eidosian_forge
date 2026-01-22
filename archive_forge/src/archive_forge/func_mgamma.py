from sympy.core.numbers import I
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.utilities.decorator import deprecated
def mgamma(mu, lower=False):
    """Returns a Dirac gamma matrix `\\gamma^\\mu` in the standard
    (Dirac) representation.

    Explanation
    ===========

    If you want `\\gamma_\\mu`, use ``gamma(mu, True)``.

    We use a convention:

    `\\gamma^5 = i \\cdot \\gamma^0 \\cdot \\gamma^1 \\cdot \\gamma^2 \\cdot \\gamma^3`

    `\\gamma_5 = i \\cdot \\gamma_0 \\cdot \\gamma_1 \\cdot \\gamma_2 \\cdot \\gamma_3 = - \\gamma^5`

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_matrices

    Examples
    ========

    >>> from sympy.physics.matrices import mgamma
    >>> mgamma(1)
    Matrix([
    [ 0,  0, 0, 1],
    [ 0,  0, 1, 0],
    [ 0, -1, 0, 0],
    [-1,  0, 0, 0]])
    """
    if mu not in (0, 1, 2, 3, 5):
        raise IndexError('Invalid Dirac index')
    if mu == 0:
        mat = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, -1, 0), (0, 0, 0, -1))
    elif mu == 1:
        mat = ((0, 0, 0, 1), (0, 0, 1, 0), (0, -1, 0, 0), (-1, 0, 0, 0))
    elif mu == 2:
        mat = ((0, 0, 0, -I), (0, 0, I, 0), (0, I, 0, 0), (-I, 0, 0, 0))
    elif mu == 3:
        mat = ((0, 0, 1, 0), (0, 0, 0, -1), (-1, 0, 0, 0), (0, 1, 0, 0))
    elif mu == 5:
        mat = ((0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 0, 0), (0, 1, 0, 0))
    m = Matrix(mat)
    if lower:
        if mu in (1, 2, 3, 5):
            m = -m
    return m
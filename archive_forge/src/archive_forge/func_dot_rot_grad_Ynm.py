from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import Function
from sympy.core.numbers import (I, Integer, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.spherical_harmonics import Ynm
from sympy.matrices.dense import zeros
from sympy.matrices.immutable import ImmutableMatrix
from sympy.utilities.misc import as_int
def dot_rot_grad_Ynm(j, p, l, m, theta, phi):
    """
    Returns dot product of rotational gradients of spherical harmonics.

    Explanation
    ===========

    This function returns the right hand side of the following expression:

    .. math ::
        \\vec{R}Y{_j^{p}} \\cdot \\vec{R}Y{_l^{m}} = (-1)^{m+p}
        \\sum\\limits_{k=|l-j|}^{l+j}Y{_k^{m+p}}  * \\alpha_{l,m,j,p,k} *
        \\frac{1}{2} (k^2-j^2-l^2+k-j-l)


    Arguments
    =========

    j, p, l, m .... indices in spherical harmonics (expressions or integers)
    theta, phi .... angle arguments in spherical harmonics

    Example
    =======

    >>> from sympy import symbols
    >>> from sympy.physics.wigner import dot_rot_grad_Ynm
    >>> theta, phi = symbols("theta phi")
    >>> dot_rot_grad_Ynm(3, 2, 2, 0, theta, phi).doit()
    3*sqrt(55)*Ynm(5, 2, theta, phi)/(11*sqrt(pi))

    """
    j = sympify(j)
    p = sympify(p)
    l = sympify(l)
    m = sympify(m)
    theta = sympify(theta)
    phi = sympify(phi)
    k = Dummy('k')

    def alpha(l, m, j, p, k):
        return sqrt((2 * l + 1) * (2 * j + 1) * (2 * k + 1) / (4 * pi)) * Wigner3j(j, l, k, S.Zero, S.Zero, S.Zero) * Wigner3j(j, l, k, p, m, -m - p)
    return S.NegativeOne ** (m + p) * Sum(Ynm(k, m + p, theta, phi) * alpha(l, m, j, p, k) / 2 * (k ** 2 - j ** 2 - l ** 2 + k - j - l), (k, abs(l - j), l + j))
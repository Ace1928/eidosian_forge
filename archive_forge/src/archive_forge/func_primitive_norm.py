import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def primitive_norm(l, alpha):
    """Compute the normalization constant for a primitive Gaussian function.

    A Gaussian function centred at the position :math:`r = (x, y, z)` is defined as

    .. math::

        G = x^{l_x} y^{l_y} z^{l_z} e^{-\\alpha r^2},

    where :math:`l = (l_x, l_y, l_z)` defines the angular momentum quantum number and :math:`\\alpha`
    is the Gaussian function exponent. The normalization constant for this function is computed as

    .. math::

        N(l, \\alpha) = (\\frac{2\\alpha}{\\pi})^{3/4} \\frac{(4 \\alpha)^{(l_x + l_y + l_z)/2}}
        {(2l_x-1)!! (2l_y-1)!! (2l_z-1)!!)^{1/2}}.

    Args:
        l (tuple[int]): angular momentum quantum number of the basis function
        alpha (array[float]): exponent of the primitive Gaussian function

    Returns:
        array[float]: normalization coefficient

    **Example**

    >>> l = (0, 0, 0)
    >>> alpha = np.array([3.425250914])
    >>> n = primitive_norm(l, alpha)
    >>> print(n)
    array([1.79444183])
    """
    lx, ly, lz = l
    n = (2 * alpha / np.pi) ** 0.75 * (4 * alpha) ** (sum(l) / 2) / qml.math.sqrt(_fac2(2 * lx - 1) * _fac2(2 * ly - 1) * _fac2(2 * lz - 1))
    return n
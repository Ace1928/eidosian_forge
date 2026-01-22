import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def gaussian_moment(li, lj, ri, rj, alpha, beta, order, r):
    """Compute the one-dimensional multipole moment integral for two primitive Gaussian functions.

    The multipole moment integral in one dimension is defined as

    .. math::

        S_{ij}^e = \\left \\langle G_i | q^e | G_j \\right \\rangle,

    where :math:`G` is a Gaussian function at dimension :math:`q = x, y, z` of the Cartesian
    coordinates system and :math:`e` is a positive integer that is represented by the ``order``
    argument. The integrals can be evaluated as
    [`Helgaker (1995) p803 <https://www.worldscientific.com/doi/abs/10.1142/9789812832115_0001>`_]

    .. math::

        S_{ij}^e = \\sum_{t=0}^{\\mathrm{min}(i+j, \\ e)} E_t^{ij} M_t^e,

    where :math:`E` and :math:`M` are the Hermite Gaussian expansion coefficient and the Hermite
    moment integral, respectively, that can be computed recursively.

    Args:
        li (integer): angular momentum for the left Gaussian function
        lj (integer): angular momentum for the right Gaussian function
        ri (float): position of the left Gaussian function
        rj (float): position of the right Gaussian function
        alpha (array[float]): exponent of the left Gaussian function
        beta (array[float]): exponent of the right Gaussian function
        order (integer): exponent of the position component
        r (array[float]): distance between the center of the Hermite Gaussian function and origin

    Returns:
        array[float]: one-dimensional multipole moment integral between primitive Gaussian functions

    **Example**

    >>> li, lj = 0, 0
    >>> ri, rj = np.array([2.0]), np.array([2.0])
    >>> alpha = np.array([3.42525091])
    >>> beta = np.array([3.42525091])
    >>> order = 1
    >>> r = 1.5
    >>> gaussian_moment(li, lj, ri, rj, alpha, beta, order, r)
    array([1.0157925])
    """
    s = 0.0
    for t in range(min(li + lj, order) + 1):
        s = s + expansion(li, lj, ri, rj, alpha, beta, t) * hermite_moment(alpha, beta, t, order, r)
    return s
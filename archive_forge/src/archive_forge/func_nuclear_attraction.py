import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def nuclear_attraction(la, lb, ra, rb, alpha, beta, r):
    """Compute nuclear attraction integral between primitive Gaussian functions.

    The nuclear attraction integral between two Gaussian functions denoted by :math:`a` and
    :math:`b` can be computed as
    [`Helgaker (1995) p820 <https://www.worldscientific.com/doi/abs/10.1142/9789812832115_0001>`_]

    .. math::

        V_{ab} = \\frac{2\\pi}{p} \\sum_{tuv} E_t^{ij} E_u^{kl} E_v^{mn} R_{tuv},

    where :math:`E` and :math:`R` represent the Hermite Gaussian expansion coefficients and the
    Hermite Coulomb integral, respectively. The sum goes over :math:`i + j + 1`, :math:`k + l + 1`
    and :math:`m + n + 1` for :math:`t`, :math:`u` and :math:`v`, respectively, and :math:`p` is
    computed from the exponents of the two Gaussian functions as :math:`p = \\alpha + \\beta`.

    Args:
        la (tuple[int]): angular momentum for the first Gaussian function
        lb (tuple[int]): angular momentum for the second Gaussian function
        ra (array[float]): position vector of the first Gaussian function
        rb (array[float]): position vector of the second Gaussian function
        alpha (array[float]): exponent of the first Gaussian function
        beta (array[float]): exponent of the second Gaussian function
        r (array[float]): position vector of nucleus

    Returns:
        array[float]: nuclear attraction integral between two Gaussian functions
    """
    l1, m1, n1 = la
    l2, m2, n2 = lb
    p = alpha + beta
    rgp = (alpha * ra[:, np.newaxis, np.newaxis] + beta * rb[:, np.newaxis, np.newaxis]) / (alpha + beta)
    dr = rgp - r[:, np.newaxis, np.newaxis]
    a = 0.0
    for t, u, v in it.product(*[range(l) for l in [l1 + l2 + 1, m1 + m2 + 1, n1 + n2 + 1]]):
        a = a + expansion(l1, l2, ra[0], rb[0], alpha, beta, t) * expansion(m1, m2, ra[1], rb[1], alpha, beta, u) * expansion(n1, n2, ra[2], rb[2], alpha, beta, v) * _hermite_coulomb(t, u, v, 0, p, dr)
    a = a * 2 * np.pi / p
    return a
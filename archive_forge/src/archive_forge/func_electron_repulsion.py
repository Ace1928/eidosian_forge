import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def electron_repulsion(la, lb, lc, ld, ra, rb, rc, rd, alpha, beta, gamma, delta):
    """Compute the electron-electron repulsion integral between four primitive Gaussian functions.

    The electron repulsion integral between four Gaussian functions denoted by :math:`a`, :math:`b`
    , :math:`c` and :math:`d` is computed as
    [`Helgaker (1995) p820 <https://www.worldscientific.com/doi/abs/10.1142/9789812832115_0001>`_]

    .. math::

        g_{abcd} = \\frac{2\\pi^{5/2}}{pq\\sqrt{p+q}} \\sum_{tuv} E_t^{o_a o_b} E_u^{m_a m_b}
        E_v^{n_a n_b} \\sum_{rsw} (-1)^{r+s+w} E_r^{o_c o_d} E_s^{m_c m_d} E_w^{n_c n_d}
        R_{t+r, u+s, v+w},

    where :math:`E` and :math:`R` are the Hermite Gaussian expansion coefficients and the
    Hermite Coulomb integral, respectively. The sums go over the angular momentum quantum numbers
    :math:`o_i + o_j + 1`, :math:`m_i + m_j + 1` and :math:`n_i + n_j + 1` respectively for
    :math:`t, u, v` and :math:`r, s, w`. The exponents of the Gaussian functions are used to compute
    :math:`p` and :math:`q` as :math:`p = \\alpha + \\beta` and :math:`q = \\gamma + \\delta`.

    Args:
        la (tuple[int]): angular momentum for the first Gaussian function
        lb (tuple[int]): angular momentum for the second Gaussian function
        lc (tuple[int]): angular momentum for the third Gaussian function
        ld (tuple[int]): angular momentum for the forth Gaussian function
        ra (array[float]): position vector of the first Gaussian function
        rb (array[float]): position vector of the second Gaussian function
        rc (array[float]): position vector of the third Gaussian function
        rd (array[float]): position vector of the forth Gaussian function
        alpha (array[float]): exponent of the first Gaussian function
        beta (array[float]): exponent of the second Gaussian function
        gamma (array[float]): exponent of the third Gaussian function
        delta (array[float]): exponent of the forth Gaussian function

    Returns:
        array[float]: electron-electron repulsion integral between four Gaussian functions
    """
    l1, m1, n1 = la
    l2, m2, n2 = lb
    l3, m3, n3 = lc
    l4, m4, n4 = ld
    p = alpha + beta
    q = gamma + delta
    p_ab = (alpha * ra[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] + beta * rb[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]) / (alpha + beta)
    p_cd = (gamma * rc[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] + delta * rd[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]) / (gamma + delta)
    ra0, ra1, ra2 = ra[0:3]
    rb0, rb1, rb2 = rb[0:3]
    rc0, rc1, rc2 = rc[0:3]
    rd0, rd1, rd2 = rd[0:3]
    g_t = [expansion(l1, l2, ra0, rb0, alpha, beta, t) for t in range(l1 + l2 + 1)]
    g_u = [expansion(m1, m2, ra1, rb1, alpha, beta, u) for u in range(m1 + m2 + 1)]
    g_v = [expansion(n1, n2, ra2, rb2, alpha, beta, v) for v in range(n1 + n2 + 1)]
    g_r = [expansion(l3, l4, rc0, rd0, gamma, delta, r) for r in range(l3 + l4 + 1)]
    g_s = [expansion(m3, m4, rc1, rd1, gamma, delta, s) for s in range(m3 + m4 + 1)]
    g_w = [expansion(n3, n4, rc2, rd2, gamma, delta, w) for w in range(n3 + n4 + 1)]
    g = 0.0
    lengths = [l1 + l2 + 1, m1 + m2 + 1, n1 + n2 + 1, l3 + l4 + 1, m3 + m4 + 1, n3 + n4 + 1]
    for t, u, v, r, s, w in it.product(*[range(length) for length in lengths]):
        g = g + g_t[t] * g_u[u] * g_v[v] * g_r[r] * g_s[s] * g_w[w] * (-1) ** (r + s + w) * _hermite_coulomb(t + r, u + s, v + w, 0, p * q / (p + q), p_ab - p_cd)
    g = g * 2 * np.pi ** 2.5 / (p * q * qml.math.sqrt(p + q))
    return g
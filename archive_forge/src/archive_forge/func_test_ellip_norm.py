import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from scipy.special._testutils import assert_func_equal
from scipy.special import ellip_harm, ellip_harm_2, ellip_normal
from scipy.integrate import IntegrationWarning
from numpy import sqrt, pi
def test_ellip_norm():

    def G01(h2, k2):
        return 4 * pi

    def G11(h2, k2):
        return 4 * pi * h2 * k2 / 3

    def G12(h2, k2):
        return 4 * pi * h2 * (k2 - h2) / 3

    def G13(h2, k2):
        return 4 * pi * k2 * (k2 - h2) / 3

    def G22(h2, k2):
        res = 2 * (h2 ** 4 + k2 ** 4) - 4 * h2 * k2 * (h2 ** 2 + k2 ** 2) + 6 * h2 ** 2 * k2 ** 2 + sqrt(h2 ** 2 + k2 ** 2 - h2 * k2) * (-2 * (h2 ** 3 + k2 ** 3) + 3 * h2 * k2 * (h2 + k2))
        return 16 * pi / 405 * res

    def G21(h2, k2):
        res = 2 * (h2 ** 4 + k2 ** 4) - 4 * h2 * k2 * (h2 ** 2 + k2 ** 2) + 6 * h2 ** 2 * k2 ** 2 + sqrt(h2 ** 2 + k2 ** 2 - h2 * k2) * (2 * (h2 ** 3 + k2 ** 3) - 3 * h2 * k2 * (h2 + k2))
        return 16 * pi / 405 * res

    def G23(h2, k2):
        return 4 * pi * h2 ** 2 * k2 * (k2 - h2) / 15

    def G24(h2, k2):
        return 4 * pi * h2 * k2 ** 2 * (k2 - h2) / 15

    def G25(h2, k2):
        return 4 * pi * h2 * k2 * (k2 - h2) ** 2 / 15

    def G32(h2, k2):
        res = 16 * (h2 ** 4 + k2 ** 4) - 36 * h2 * k2 * (h2 ** 2 + k2 ** 2) + 46 * h2 ** 2 * k2 ** 2 + sqrt(4 * (h2 ** 2 + k2 ** 2) - 7 * h2 * k2) * (-8 * (h2 ** 3 + k2 ** 3) + 11 * h2 * k2 * (h2 + k2))
        return 16 * pi / 13125 * k2 * h2 * res

    def G31(h2, k2):
        res = 16 * (h2 ** 4 + k2 ** 4) - 36 * h2 * k2 * (h2 ** 2 + k2 ** 2) + 46 * h2 ** 2 * k2 ** 2 + sqrt(4 * (h2 ** 2 + k2 ** 2) - 7 * h2 * k2) * (8 * (h2 ** 3 + k2 ** 3) - 11 * h2 * k2 * (h2 + k2))
        return 16 * pi / 13125 * h2 * k2 * res

    def G34(h2, k2):
        res = 6 * h2 ** 4 + 16 * k2 ** 4 - 12 * h2 ** 3 * k2 - 28 * h2 * k2 ** 3 + 34 * h2 ** 2 * k2 ** 2 + sqrt(h2 ** 2 + 4 * k2 ** 2 - h2 * k2) * (-6 * h2 ** 3 - 8 * k2 ** 3 + 9 * h2 ** 2 * k2 + 13 * h2 * k2 ** 2)
        return 16 * pi / 13125 * h2 * (k2 - h2) * res

    def G33(h2, k2):
        res = 6 * h2 ** 4 + 16 * k2 ** 4 - 12 * h2 ** 3 * k2 - 28 * h2 * k2 ** 3 + 34 * h2 ** 2 * k2 ** 2 + sqrt(h2 ** 2 + 4 * k2 ** 2 - h2 * k2) * (6 * h2 ** 3 + 8 * k2 ** 3 - 9 * h2 ** 2 * k2 - 13 * h2 * k2 ** 2)
        return 16 * pi / 13125 * h2 * (k2 - h2) * res

    def G36(h2, k2):
        res = 16 * h2 ** 4 + 6 * k2 ** 4 - 28 * h2 ** 3 * k2 - 12 * h2 * k2 ** 3 + 34 * h2 ** 2 * k2 ** 2 + sqrt(4 * h2 ** 2 + k2 ** 2 - h2 * k2) * (-8 * h2 ** 3 - 6 * k2 ** 3 + 13 * h2 ** 2 * k2 + 9 * h2 * k2 ** 2)
        return 16 * pi / 13125 * k2 * (k2 - h2) * res

    def G35(h2, k2):
        res = 16 * h2 ** 4 + 6 * k2 ** 4 - 28 * h2 ** 3 * k2 - 12 * h2 * k2 ** 3 + 34 * h2 ** 2 * k2 ** 2 + sqrt(4 * h2 ** 2 + k2 ** 2 - h2 * k2) * (8 * h2 ** 3 + 6 * k2 ** 3 - 13 * h2 ** 2 * k2 - 9 * h2 * k2 ** 2)
        return 16 * pi / 13125 * k2 * (k2 - h2) * res

    def G37(h2, k2):
        return 4 * pi * h2 ** 2 * k2 ** 2 * (k2 - h2) ** 2 / 105
    known_funcs = {(0, 1): G01, (1, 1): G11, (1, 2): G12, (1, 3): G13, (2, 1): G21, (2, 2): G22, (2, 3): G23, (2, 4): G24, (2, 5): G25, (3, 1): G31, (3, 2): G32, (3, 3): G33, (3, 4): G34, (3, 5): G35, (3, 6): G36, (3, 7): G37}

    def _ellip_norm(n, p, h2, k2):
        func = known_funcs[n, p]
        return func(h2, k2)
    _ellip_norm = np.vectorize(_ellip_norm)

    def ellip_normal_known(h2, k2, n, p):
        return _ellip_norm(n, p, h2, k2)
    np.random.seed(1234)
    h2 = np.random.pareto(0.5, size=1)
    k2 = h2 * (1 + np.random.pareto(0.5, size=h2.size))
    points = []
    for n in range(4):
        for p in range(1, 2 * n + 2):
            points.append((h2, k2, np.full(h2.size, n), np.full(h2.size, p)))
    points = np.array(points)
    with suppress_warnings() as sup:
        sup.filter(IntegrationWarning, 'The occurrence of roundoff error')
        assert_func_equal(ellip_normal, ellip_normal_known, points, rtol=1e-12)
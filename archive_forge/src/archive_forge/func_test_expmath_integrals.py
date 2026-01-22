import pytest
from mpmath import *
def test_expmath_integrals():
    for prec in [15, 30, 50]:
        mp.dps = prec
        assert ae(quadts(lambda x: x / sinh(x), [0, inf]), pi ** 2 / 4)
        assert ae(quadts(lambda x: log(x) ** 2 / (1 + x ** 2), [0, inf]), pi ** 3 / 8)
        assert ae(quadts(lambda x: (1 + x ** 2) / (1 + x ** 4), [0, inf]), pi / sqrt(2))
        assert ae(quadts(lambda x: log(x) / cosh(x) ** 2, [0, inf]), log(pi) - 2 * log(2) - euler)
        assert ae(quadts(lambda x: log(1 + x ** 3) / (1 - x + x ** 2), [0, inf]), 2 * pi * log(3) / sqrt(3))
        assert ae(quadts(lambda x: log(x) ** 2 / (x ** 2 + x + 1), [0, 1]), 8 * pi ** 3 / (81 * sqrt(3)))
        assert ae(quadts(lambda x: log(cos(x)) ** 2, [0, pi / 2]), pi / 2 * (log(2) ** 2 + pi ** 2 / 12))
        assert ae(quadts(lambda x: x ** 2 / sin(x) ** 2, [0, pi / 2]), pi * log(2))
        assert ae(quadts(lambda x: x ** 2 / sqrt(exp(x) - 1), [0, inf]), 4 * pi * (log(2) ** 2 + pi ** 2 / 12))
        assert ae(quadts(lambda x: x * exp(-x) * sqrt(1 - exp(-2 * x)), [0, inf]), pi * (1 + 2 * log(2)) / 8)
    mp.dps = 15
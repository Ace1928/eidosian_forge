import pytest
from mpmath import *
def test_basic_integrals():
    for prec in [15, 30, 100]:
        mp.dps = prec
        assert ae(quadts(lambda x: x ** 3 - 3 * x ** 2, [-2, 4]), -12)
        assert ae(quadgl(lambda x: x ** 3 - 3 * x ** 2, [-2, 4]), -12)
        assert ae(quadts(sin, [0, pi]), 2)
        assert ae(quadts(sin, [0, 2 * pi]), 0)
        assert ae(quadts(exp, [-inf, -1]), 1 / e)
        assert ae(quadts(lambda x: exp(-x), [0, inf]), 1)
        assert ae(quadts(lambda x: exp(-x * x), [-inf, inf]), sqrt(pi))
        assert ae(quadts(lambda x: 1 / (1 + x * x), [-1, 1]), pi / 2)
        assert ae(quadts(lambda x: 1 / (1 + x * x), [-inf, inf]), pi)
        assert ae(quadts(lambda x: 2 * sqrt(1 - x * x), [-1, 1]), pi)
    mp.dps = 15
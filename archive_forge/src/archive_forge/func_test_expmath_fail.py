import pytest
from mpmath import *
@pytest.mark.xfail
def test_expmath_fail():
    assert ae(quadts(lambda x: sqrt(tan(x)), [0, pi / 2]), pi * sqrt(2) / 2)
    assert ae(quadts(lambda x: atan(x) / (x * sqrt(1 - x ** 2)), [0, 1]), pi * log(1 + sqrt(2)) / 2)
    assert ae(quadts(lambda x: log(1 + x ** 2) / x ** 2, [0, 1]), pi / 2 - log(2))
    assert ae(quadts(lambda x: x ** 2 / ((1 + x ** 4) * sqrt(1 - x ** 4)), [0, 1]), pi / 8)
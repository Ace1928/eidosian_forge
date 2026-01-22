import random
from mpmath import *
from mpmath.libmp import *
def test_convert_rational():
    mp.dps = 15
    assert from_rational(30, 5, 53, round_nearest) == (0, 3, 1, 2)
    assert from_rational(-7, 4, 53, round_nearest) == (1, 7, -2, 3)
    assert to_rational((0, 1, -1, 1)) == (1, 2)
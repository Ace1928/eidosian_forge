from mpmath import *
from mpmath.libmp import *
def test_rounding_bugs():
    assert from_man_exp(72057594037927935, -56, 53, round_up) == (0, 1, 0, 1)
    assert from_man_exp(73786976294838205979, -65, 53, round_nearest) == (0, 1, 1, 1)
    assert from_man_exp(31, 0, 4, round_up) == (0, 1, 5, 1)
    assert from_man_exp(-31, 0, 4, round_floor) == (1, 1, 5, 1)
    assert from_man_exp(255, 0, 7, round_up) == (0, 1, 8, 1)
    assert from_man_exp(-255, 0, 7, round_floor) == (1, 1, 8, 1)
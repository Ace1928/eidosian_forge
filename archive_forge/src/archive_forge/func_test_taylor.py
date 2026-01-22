from mpmath import *
def test_taylor():
    mp.dps = 15
    assert taylor(sqrt, 1, 4) == [1, 0.5, -0.125, 0.0625, -0.0390625]
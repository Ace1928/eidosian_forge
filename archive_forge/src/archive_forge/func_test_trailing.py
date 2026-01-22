from mpmath import *
from mpmath.libmp import *
def test_trailing():
    assert trailing(0) == 0
    assert trailing(1) == 0
    assert trailing(2) == 1
    assert trailing(7) == 0
    assert trailing(8) == 3
    assert trailing(2 ** 100) == 100
    assert trailing(2 ** 100 - 1) == 0
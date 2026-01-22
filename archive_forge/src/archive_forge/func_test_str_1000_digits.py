import mpmath
from mpmath import *
from mpmath.libmp import *
import random
import sys
def test_str_1000_digits():
    mp.dps = 1001
    assert str(mpf(2) ** 0.5)[-10:-1] == '9518488472'[:9]
    assert str(pi)[-10:-1] == '2164201989'[:9]
    mp.dps = 15
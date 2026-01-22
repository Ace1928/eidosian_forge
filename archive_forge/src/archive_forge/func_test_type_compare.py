import mpmath
from mpmath import *
from mpmath.libmp import *
import random
import sys
def test_type_compare():
    assert mpf(2) == mpc(2, 0)
    assert mpf(0) == mpc(0)
    assert mpf(2) != mpc(2, 1e-05)
    assert mpf(2) == 2.0
    assert mpf(2) != 3.0
    assert mpf(2) == 2
    assert mpf(2) != '2.0'
    assert mpc(2) != '2.0'
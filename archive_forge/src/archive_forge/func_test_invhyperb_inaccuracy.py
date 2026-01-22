from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_invhyperb_inaccuracy():
    mp.dps = 15
    assert (asinh(1e-05) * 10 ** 5).ae(0.9999999999833333)
    assert (asinh(1e-10) * 10 ** 10).ae(1)
    assert (asinh(1e-50) * 10 ** 50).ae(1)
    assert (asinh(-1e-05) * 10 ** 5).ae(-0.9999999999833333)
    assert (asinh(-1e-10) * 10 ** 10).ae(-1)
    assert (asinh(-1e-50) * 10 ** 50).ae(-1)
    assert asinh(10 ** 20).ae(46.74484904044086)
    assert asinh(-10 ** 20).ae(-46.74484904044086)
    assert (tanh(1e-10) * 10 ** 10).ae(1)
    assert (tanh(-1e-10) * 10 ** 10).ae(-1)
    assert (atanh(1e-10) * 10 ** 10).ae(1)
    assert (atanh(-1e-10) * 10 ** 10).ae(-1)
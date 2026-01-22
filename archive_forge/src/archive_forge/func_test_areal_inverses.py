from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_areal_inverses():
    assert asin(mpf(0)) == 0
    assert asinh(mpf(0)) == 0
    assert acosh(mpf(1)) == 0
    assert isinstance(asin(mpf(0.5)), mpf)
    assert isinstance(asin(mpf(2.0)), mpc)
    assert isinstance(acos(mpf(0.5)), mpf)
    assert isinstance(acos(mpf(2.0)), mpc)
    assert isinstance(atanh(mpf(0.1)), mpf)
    assert isinstance(atanh(mpf(1.1)), mpc)
    random.seed(1)
    for i in range(50):
        x = random.uniform(0, 1)
        assert asin(mpf(x)).ae(math.asin(x))
        assert acos(mpf(x)).ae(math.acos(x))
        x = random.uniform(-10, 10)
        assert asinh(mpf(x)).ae(cmath.asinh(x).real)
        assert isinstance(asinh(mpf(x)), mpf)
        x = random.uniform(1, 10)
        assert acosh(mpf(x)).ae(cmath.acosh(x).real)
        assert isinstance(acosh(mpf(x)), mpf)
        x = random.uniform(-10, 0.999)
        assert isinstance(acosh(mpf(x)), mpc)
        x = random.uniform(-1, 1)
        assert atanh(mpf(x)).ae(cmath.atanh(x).real)
        assert isinstance(atanh(mpf(x)), mpf)
    dps = mp.dps
    mp.dps = 300
    assert isinstance(asin(0.5), mpf)
    mp.dps = 1000
    assert asin(1).ae(pi / 2)
    assert asin(-1).ae(-pi / 2)
    mp.dps = dps
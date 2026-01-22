from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_mpcfun_real_imag():
    mp.dps = 15
    x = mpf(0.3)
    y = mpf(0.4)
    assert exp(mpc(x, 0)) == exp(x)
    assert exp(mpc(0, y)) == mpc(cos(y), sin(y))
    assert cos(mpc(x, 0)) == cos(x)
    assert sin(mpc(x, 0)) == sin(x)
    assert cos(mpc(0, y)) == cosh(y)
    assert sin(mpc(0, y)) == mpc(0, sinh(y))
    assert cospi(mpc(x, 0)) == cospi(x)
    assert sinpi(mpc(x, 0)) == sinpi(x)
    assert cospi(mpc(0, y)).ae(cosh(pi * y))
    assert sinpi(mpc(0, y)).ae(mpc(0, sinh(pi * y)))
    c, s = cospi_sinpi(mpc(x, 0))
    assert c == cospi(x)
    assert s == sinpi(x)
    c, s = cospi_sinpi(mpc(0, y))
    assert c.ae(cosh(pi * y))
    assert s.ae(mpc(0, sinh(pi * y)))
    c, s = cos_sin(mpc(x, 0))
    assert c == cos(x)
    assert s == sin(x)
    c, s = cos_sin(mpc(0, y))
    assert c == cosh(y)
    assert s == mpc(0, sinh(y))
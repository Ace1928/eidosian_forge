from mpmath import *
from random import seed, randint, random
import math
def test_double_compatibility():
    mp.prec = 53
    for x, y in zip(xs, ys):
        mpx = mpf(x)
        mpy = mpf(y)
        assert mpf(x) == x
        assert (mpx < mpy) == (x < y)
        assert (mpx > mpy) == (x > y)
        assert (mpx == mpy) == (x == y)
        assert (mpx != mpy) == (x != y)
        assert (mpx <= mpy) == (x <= y)
        assert (mpx >= mpy) == (x >= y)
        assert mpx == mpx
        if uses_x87:
            mp.prec = 64
            a = mpx + mpy
            b = mpx * mpy
            c = mpx / mpy
            d = mpx % mpy
            mp.prec = 53
            assert +a == x + y
            assert +b == x * y
            assert +c == x / y
            assert +d == x % y
        else:
            assert mpx + mpy == x + y
            assert mpx * mpy == x * y
            assert mpx / mpy == x / y
            assert mpx % mpy == x % y
        assert abs(mpx) == abs(x)
        assert mpf(repr(x)) == x
        assert ceil(mpx) == math.ceil(x)
        assert floor(mpx) == math.floor(x)
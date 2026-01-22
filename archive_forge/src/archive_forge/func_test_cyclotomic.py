from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_cyclotomic():
    mp.dps = 15
    assert [cyclotomic(n, 1) for n in range(31)] == [1, 0, 2, 3, 2, 5, 1, 7, 2, 3, 1, 11, 1, 13, 1, 1, 2, 17, 1, 19, 1, 1, 1, 23, 1, 5, 1, 3, 1, 29, 1]
    assert [cyclotomic(n, -1) for n in range(31)] == [1, -2, 0, 1, 2, 1, 3, 1, 2, 1, 5, 1, 1, 1, 7, 1, 2, 1, 3, 1, 1, 1, 11, 1, 1, 1, 13, 1, 1, 1, 1]
    assert [cyclotomic(n, j) for n in range(21)] == [1, -1 + j, 1 + j, j, 0, 1, -j, j, 2, -j, 1, j, 3, 1, -j, 1, 2, 1, j, j, 5]
    assert [cyclotomic(n, -j) for n in range(21)] == [1, -1 - j, 1 - j, -j, 0, 1, j, -j, 2, j, 1, -j, 3, 1, j, 1, 2, 1, -j, -j, 5]
    assert cyclotomic(1624, j) == 1
    assert cyclotomic(33600, j) == 1
    u = sqrt(j, prec=500)
    assert cyclotomic(8, u).ae(0)
    assert cyclotomic(30, u).ae(5.82842712474619)
    assert cyclotomic(2040, u).ae(1)
    assert cyclotomic(0, 2.5) == 1
    assert cyclotomic(1, 2.5) == 2.5 - 1
    assert cyclotomic(2, 2.5) == 2.5 + 1
    assert cyclotomic(3, 2.5) == 2.5 ** 2 + 2.5 + 1
    assert cyclotomic(7, 2.5) == 406.234375
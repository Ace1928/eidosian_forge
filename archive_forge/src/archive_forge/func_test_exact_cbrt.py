from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_exact_cbrt():
    for i in range(0, 20000, 200):
        assert cbrt(mpf(i * i * i)) == i
    random.seed(1)
    for prec in [100, 300, 1000, 10000]:
        mp.dps = prec
        A = random.randint(10 ** (prec // 2 - 2), 10 ** (prec // 2 - 1))
        assert cbrt(mpf(A * A * A)) == A
    mp.dps = 15
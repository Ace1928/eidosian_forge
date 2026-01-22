from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_complex_sqrt_accuracy():

    def test_mpc_sqrt(lst):
        for a, b in lst:
            z = mpc(a + j * b)
            assert mpc_ae(sqrt(z * z), z)
            z = mpc(-a + j * b)
            assert mpc_ae(sqrt(z * z), -z)
            z = mpc(a - j * b)
            assert mpc_ae(sqrt(z * z), z)
            z = mpc(-a - j * b)
            assert mpc_ae(sqrt(z * z), -z)
    random.seed(2)
    N = 10
    mp.dps = 30
    dps = mp.dps
    test_mpc_sqrt([(random.uniform(0, 10), random.uniform(0, 10)) for i in range(N)])
    test_mpc_sqrt([(i + 0.1, (i + 0.2) * 10 ** i) for i in range(N)])
    mp.dps = 15
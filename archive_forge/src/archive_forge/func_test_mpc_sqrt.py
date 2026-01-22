from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
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
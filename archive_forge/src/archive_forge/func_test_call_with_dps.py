from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_call_with_dps():
    mp.dps = 15
    assert abs(exp(1, dps=30) - e(dps=35)) < 1e-29
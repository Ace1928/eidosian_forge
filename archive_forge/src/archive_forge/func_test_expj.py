from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_expj():
    assert expj(0) == 1
    assert expj(1).ae(exp(j))
    assert expj(j).ae(exp(-1))
    assert expj(1 + j).ae(exp(j * (1 + j)))
    assert expjpi(0) == 1
    assert expjpi(1).ae(exp(j * pi))
    assert expjpi(j).ae(exp(-pi))
    assert expjpi(1 + j).ae(exp(j * pi * (1 + j)))
    assert expjpi(-10 ** 15 * j).ae('2.22579818340535731e+1364376353841841')
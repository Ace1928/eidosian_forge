import numpy as np
import math
from numba import cuda
from numba.types import float32, float64, int32, void
from numba.cuda.testing import unittest, CUDATestCase
def simple_frexp(aryx, aryexp, arg):
    aryx[0], aryexp[0] = math.frexp(arg)
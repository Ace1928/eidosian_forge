import numpy as np
from numba import vectorize
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
@vectorize(['complex128(complex128)'], target='cuda')
def vcomp(a):
    return a * a + 1.0
import numpy as np
import numba
from numba.tests.support import TestCase
def call_f(self):
    a = np.zeros(10)
    for f in self.f:
        f(a)
    return a
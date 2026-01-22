import os
import sys
import time
from itertools import zip_longest
import numpy as np
from numpy.testing import assert_
import pytest
from scipy.special._testutils import assert_func_equal
def pytype(x):
    if abs(x.imag) > 1e-16 * (1 + abs(x.real)):
        return np.nan
    else:
        return mpf2float(x.real)
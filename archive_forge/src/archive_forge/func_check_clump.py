import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def check_clump(self, f):
    for i in range(1, 7):
        for j in range(2 ** i):
            k = np.arange(i, dtype=int)
            ja = np.full(i, j, dtype=int)
            a = masked_array(2 ** k)
            a.mask = ja & 2 ** k != 0
            s = 0
            for sl in f(a):
                s += a.data[sl].sum()
            if f == clump_unmasked:
                assert_equal(a.compressed().sum(), s)
            else:
                a.mask = ~a.mask
                assert_equal(a.compressed().sum(), s)
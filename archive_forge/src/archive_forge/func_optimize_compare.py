import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def optimize_compare(self, subscripts, operands=None):
    if operands is None:
        args = [subscripts]
        terms = subscripts.split('->')[0].split(',')
        for term in terms:
            dims = [global_size_dict[x] for x in term]
            args.append(np.random.rand(*dims))
    else:
        args = [subscripts] + operands
    noopt = np.einsum(*args, optimize=False)
    opt = np.einsum(*args, optimize='greedy')
    assert_almost_equal(opt, noopt)
    opt = np.einsum(*args, optimize='optimal')
    assert_almost_equal(opt, noopt)
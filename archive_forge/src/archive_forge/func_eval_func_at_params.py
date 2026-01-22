import os
import functools
import operator
from scipy._lib import _pep440
import numpy as np
from numpy.testing import assert_
import pytest
import scipy.special as sc
def eval_func_at_params(func, skip_mask=None):
    if self.vectorized:
        got = func(*params)
    else:
        got = []
        for j in range(len(params[0])):
            if skip_mask is not None and skip_mask[j]:
                got.append(np.nan)
                continue
            got.append(func(*tuple([params[i][j] for i in range(len(params))])))
        got = np.asarray(got)
    if not isinstance(got, tuple):
        got = (got,)
    return got
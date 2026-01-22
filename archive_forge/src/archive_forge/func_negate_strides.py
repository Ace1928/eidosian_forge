import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def negate_strides(arrs):
    ret = []
    for a in arrs:
        b = np.zeros_like(a)
        if b.ndim == 2:
            b = b[::-1, ::-1]
        elif b.ndim == 1:
            b = b[::-1]
        else:
            raise ValueError('negate_strides only works for ndim = 1 or 2 arrays')
        b[...] = a
        ret.append(b)
    return ret
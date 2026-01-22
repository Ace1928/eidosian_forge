import scipy.linalg.interpolative as pymatrixid
import numpy as np
from scipy.linalg import hilbert, svdvals, norm
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg.interpolative import interp_decomp
from numpy.testing import (assert_, assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
import sys
def test_badcall(self):
    A = hilbert(5).astype(np.float32)
    with assert_raises(ValueError):
        pymatrixid.interp_decomp(A, 1e-06, rand=False)
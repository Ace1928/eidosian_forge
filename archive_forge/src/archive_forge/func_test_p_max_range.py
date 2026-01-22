from functools import partial
from itertools import product
import numpy as np
import pytest
from numpy.testing import (assert_allclose, assert_, assert_equal,
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import aslinearoperator
import scipy.linalg
from scipy.sparse.linalg import expm as sp_expm
from scipy.sparse.linalg._expm_multiply import (_theta, _compute_p_max,
from scipy._lib._util import np_long
def test_p_max_range(self):
    for m_max in range(1, 55 + 1):
        p_max = _compute_p_max(m_max)
        assert_(p_max * (p_max - 1) <= m_max + 1)
        p_too_big = p_max + 1
        assert_(p_too_big * (p_too_big - 1) > m_max + 1)
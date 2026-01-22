from __future__ import division
import pytest
from mpmath import *
def test_LU_cache():
    A = randmatrix(3)
    LU = LU_decomp(A)
    assert A._LU == LU_decomp(A)
    A[0, 0] = -1000
    assert A._LU is None
import pytest
import sys
from mpmath import *
def test_matrix_creation():
    assert diag([1, 2, 3]) == matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    A1 = ones(2, 3)
    assert A1.rows == 2 and A1.cols == 3
    for a in A1:
        assert a == 1
    A2 = zeros(3, 2)
    assert A2.rows == 3 and A2.cols == 2
    for a in A2:
        assert a == 0
    assert randmatrix(10) != randmatrix(10)
    one = mpf(1)
    assert hilbert(3) == matrix([[one, one / 2, one / 3], [one / 2, one / 3, one / 4], [one / 3, one / 4, one / 5]])
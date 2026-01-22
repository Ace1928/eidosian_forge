import pytest
import sys
from mpmath import *
def test_matrix_slices():
    A = matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    V = matrix([1, 2, 3, 4, 5])
    assert A[:, :] == A
    assert A[:, 1] == matrix([[2], [5], [8]])
    assert A[2, :] == matrix([[7, 8, 9]])
    assert A[1:3, 1:3] == matrix([[5, 6], [8, 9]])
    assert V[2:4] == matrix([3, 4])
    pytest.raises(IndexError, lambda: A[:, 1:6])
    A1 = matrix(3)
    A1[:, :] = A
    assert A1[:, :] == matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    A1[0, :] = matrix([[10, 11, 12]])
    assert A1 == matrix([[10, 11, 12], [4, 5, 6], [7, 8, 9]])
    A1[:, 2] = matrix([[13], [14], [15]])
    assert A1 == matrix([[10, 11, 13], [4, 5, 14], [7, 8, 15]])
    A1[:2, :2] = matrix([[16, 17], [18, 19]])
    assert A1 == matrix([[16, 17, 13], [18, 19, 14], [7, 8, 15]])
    V[1:3] = 10
    assert V == matrix([1, 10, 10, 4, 5])
    with pytest.raises(ValueError):
        A1[2, :] = A[:, 1]
    with pytest.raises(IndexError):
        A1[2, 1:20] = A[:, :]
    A1[:, 2] = 10
    assert A1 == matrix([[16, 17, 10], [18, 19, 10], [7, 8, 10]])
    A1[:, :] = 40
    for x in A1:
        assert x == 40
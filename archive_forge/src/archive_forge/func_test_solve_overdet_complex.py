from __future__ import division
import pytest
from mpmath import *
def test_solve_overdet_complex():
    A = matrix([[1, 2j], [3, 4j], [5, 6]])
    b = matrix([1 + j, 2, -j])
    assert norm(residual(A, lu_solve(A, b), b)) < 1.0208
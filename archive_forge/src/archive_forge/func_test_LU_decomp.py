from __future__ import division
import pytest
from mpmath import *
def test_LU_decomp():
    A = A3.copy()
    b = b3
    A, p = LU_decomp(A)
    y = L_solve(A, b, p)
    x = U_solve(A, y)
    assert p == [2, 1, 2, 3]
    assert [round(i, 14) for i in x] == [3.78953107960742, 2.99890948745911, -0.08178844056707, 3.87131952017448, 2.91712104689204]
    A = A4.copy()
    b = b4
    A, p = LU_decomp(A)
    y = L_solve(A, b, p)
    x = U_solve(A, y)
    assert p == [0, 3, 4, 3]
    assert [round(i, 14) for i in x] == [2.63836258996192, 2.66438344623684, 0.79208015947959, -2.50883764541019, -1.0567657691375]
    A = randmatrix(3)
    bak = A.copy()
    LU_decomp(A, overwrite=1)
    assert A != bak
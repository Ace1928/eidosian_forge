import pytest
import sys
from mpmath import *
def test_interval_matrix_scalar_mult():
    """Multiplication of iv.matrix and any scalar type"""
    a = mpi(-1, 1)
    b = a + a * 2j
    c = mpf(42)
    d = c + c * 2j
    e = 1.234
    f = fp.convert(e)
    g = e + e * 3j
    h = fp.convert(g)
    M = iv.ones(1)
    for x in [a, b, c, d, e, f, g, h]:
        assert x * M == iv.matrix([x])
        assert M * x == iv.matrix([x])
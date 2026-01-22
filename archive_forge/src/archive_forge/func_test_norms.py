import pytest
import sys
from mpmath import *
def test_norms():
    A = matrix([[1, -2], [-3, -1], [2, 1]])
    assert mnorm(A, 1) == 6
    assert mnorm(A, inf) == 4
    assert mnorm(A, 'F') == sqrt(20)
    assert norm(-3) == 3
    x = [1, -2, 7, -12]
    assert norm(x, 1) == 22
    assert round(norm(x, 2), 10) == 14.0712472795
    assert round(norm(x, 10), 10) == 12.0054633727
    assert norm(x, inf) == 12
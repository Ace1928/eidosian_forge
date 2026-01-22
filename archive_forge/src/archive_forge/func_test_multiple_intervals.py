import pytest
from mpmath import *
def test_multiple_intervals():
    y, err = quad(lambda x: sign(x), [-0.5, 0.9, 1], maxdegree=2, error=True)
    assert abs(y - 0.5) < 2 * err
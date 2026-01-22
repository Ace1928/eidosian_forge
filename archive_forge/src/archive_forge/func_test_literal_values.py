import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
import scipy.special as sc
def test_literal_values(self):
    y = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    actual = sc.erfinv(y)
    expected = [0.0, 0.08885599049425769, 0.1791434546212917, 0.2724627147267543, 0.37080715859355795, 0.4769362762044699, 0.5951160814499948, 0.7328690779592167, 0.9061938024368233, 1.1630871536766743]
    assert_allclose(actual, expected, rtol=0, atol=1e-15)
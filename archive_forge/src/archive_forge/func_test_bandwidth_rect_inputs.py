import numpy as np
from scipy.linalg import bandwidth, issymmetric, ishermitian
import pytest
from pytest import raises
@pytest.mark.parametrize('T', [x for x in np.typecodes['All'] if x not in 'eGUVOMm'])
def test_bandwidth_rect_inputs(T):
    n, m = (10, 20)
    k = 5
    R = np.zeros([n, m], dtype=T, order='F')
    R[[x for x in range(n)], [x for x in range(n)]] = 1
    R[[x for x in range(n - k)], [x for x in range(k, n)]] = 1
    R[[x for x in range(1, n)], [x for x in range(n - 1)]] = 1
    R[[x for x in range(k, n)], [x for x in range(n - k)]] = 1
    assert bandwidth(R) == (k, k)
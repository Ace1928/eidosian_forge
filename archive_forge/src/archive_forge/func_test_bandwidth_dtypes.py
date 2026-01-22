import numpy as np
from scipy.linalg import bandwidth, issymmetric, ishermitian
import pytest
from pytest import raises
def test_bandwidth_dtypes():
    n = 5
    for t in np.typecodes['All']:
        A = np.zeros([n, n], dtype=t)
        if t in 'eUVOMm':
            raises(TypeError, bandwidth, A)
        elif t == 'G':
            pass
        else:
            _ = bandwidth(A)
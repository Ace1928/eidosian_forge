import numpy as np
from scipy.linalg import bandwidth, issymmetric, ishermitian
import pytest
from pytest import raises
def test_issymmetric_approximate_results():
    n = 20
    rng = np.random.RandomState(123456789)
    x = rng.uniform(high=5.0, size=[n, n])
    y = x @ x.T
    p = rng.standard_normal([n, n])
    z = p @ y @ p.T
    assert issymmetric(z, atol=1e-10)
    assert issymmetric(z, atol=1e-10, rtol=0.0)
    assert issymmetric(z, atol=0.0, rtol=1e-12)
    assert issymmetric(z, atol=1e-13, rtol=1e-12)
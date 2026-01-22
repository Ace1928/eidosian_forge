import numpy as np
from numpy.testing import assert_allclose, assert_
from scipy.special._testutils import FuncData
from scipy.special import gamma, gammaln, loggamma
def test_identities1():
    x = np.array([-99.5, -9.5, -0.5, 0.5, 9.5, 99.5])
    y = x.copy()
    x, y = np.meshgrid(x, y)
    z = (x + 1j * y).flatten()
    dataset = np.vstack((z, gamma(z))).T

    def f(z):
        return np.exp(loggamma(z))
    FuncData(f, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()
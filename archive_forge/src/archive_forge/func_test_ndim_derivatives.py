import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_ndim_derivatives(self):
    poly1 = self.true_poly
    poly2 = np.polynomial.Polynomial([-2, 5, 3, -1])
    poly3 = np.polynomial.Polynomial([12, -3, 4, -5, 6])
    ys = np.stack((poly1(self.xs), poly2(self.xs), poly3(self.xs)), axis=-1)
    P = BarycentricInterpolator(self.xs, ys, axis=0)
    D = P.derivatives(self.test_xs)
    for i in range(D.shape[0]):
        assert_allclose(D[i], np.stack((poly1.deriv(i)(self.test_xs), poly2.deriv(i)(self.test_xs), poly3.deriv(i)(self.test_xs)), axis=-1), atol=1e-12)
import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_lagrange(self):
    P = BarycentricInterpolator(self.xs, self.ys)
    assert_allclose(P(self.test_xs), self.true_poly(self.test_xs))
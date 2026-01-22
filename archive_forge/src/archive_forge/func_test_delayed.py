import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_delayed(self):
    P = BarycentricInterpolator(self.xs)
    P.set_yi(self.ys)
    assert_almost_equal(self.true_poly(self.test_xs), P(self.test_xs))
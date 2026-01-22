import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
@pytest.mark.parametrize('method', quantile_methods)
@pytest.mark.parametrize('alpha', [0.2, 0.5, 0.9])
def test_quantile_identification_equation(self, method, alpha):
    rng = np.random.default_rng(4321)
    n = 102
    y = rng.random(n)
    x = np.quantile(y, alpha, method=method)
    if method in ('higher',):
        assert np.abs(np.mean(self.V(x, y, alpha))) > 0.1 / n
    elif int(n * alpha) == n * alpha:
        assert_allclose(np.mean(self.V(x, y, alpha)), 0, atol=1e-14)
    else:
        assert_allclose(np.mean(self.V(x, y, alpha)), 0, atol=1 / n / np.amin([alpha, 1 - alpha]))
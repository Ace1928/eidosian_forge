from . import util
import numpy as np
import pytest
from numpy.testing import assert_allclose
def test_bindc_add_arr(self):
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    out = self.module.coddity.add_arr(a, b)
    exp_out = a * 2
    assert_allclose(out, exp_out)
import pytest
import numpy as np
from numpy.testing import assert_allclose
import scipy.special as sc
@pytest.mark.parametrize('result', [sc.expi(complex(-1, 0)), sc.expi(complex(-1, -0.0)), sc.expi(-1)])
def test_branch_cut(self, result):
    desired = -0.21938393439552029
    assert_allclose(result, desired, atol=0, rtol=1e-14)
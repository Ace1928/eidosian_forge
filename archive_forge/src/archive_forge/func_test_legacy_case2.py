import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import scipy.special as sc
def test_legacy_case2(self):
    assert sc.hyp1f1(-4, -3, 0) == np.inf
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import scipy.special as sc
from scipy.special._testutils import FuncData
def test_infinite_limits(self):
    assert sc.gammaincc(1000, 100) == sc.gammaincc(np.inf, 100)
    assert_allclose(sc.gammaincc(100, 1000), sc.gammaincc(100, np.inf), atol=1e-200, rtol=0)
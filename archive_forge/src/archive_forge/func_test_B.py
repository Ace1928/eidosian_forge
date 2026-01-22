from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
import statsmodels.datasets.macrodata
from statsmodels.tsa.vector_ar.svar_model import SVAR
def test_B(self):
    assert_almost_equal(np.abs(self.res1.B), self.res2.B, DECIMAL_4)
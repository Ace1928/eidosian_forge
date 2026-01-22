import numpy as np
from numpy.testing import assert_allclose, assert_array_less
from scipy import stats
import pytest
import statsmodels.nonparametric.kernels_asymmetric as kern
@pytest.mark.parametrize('case', kernels_unit)
def test_kernels(self, case):
    super().test_kernels(case)
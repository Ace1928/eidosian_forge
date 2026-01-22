import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from statsmodels.datasets import elnino
from statsmodels.graphics.functional import (
@pytest.mark.slow
@pytest.mark.matplotlib
def test_hdr_ncomp(close_figures):
    try:
        _, hdr = hdrboxplot(data, ncomp=3, seed=12345)
        median_t = [24.33, 25.71, 26.04, 25.08, 23.74, 22.4, 21.32, 20.45, 20.25, 20.53, 21.2, 22.39]
        assert_almost_equal(hdr.median, median_t, decimal=2)
    except OSError:
        pytest.xfail('Multiprocess randomly crashes in Windows testing')
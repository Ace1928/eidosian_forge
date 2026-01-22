import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from statsmodels.datasets import elnino
from statsmodels.graphics.functional import (
@pytest.mark.matplotlib
def test_hdr_bw(close_figures):
    try:
        _, hdr = hdrboxplot(data, bw='cv_ml', seed=12345)
        median_t = [24.25, 25.64, 25.99, 25.04, 23.71, 22.38, 21.31, 20.44, 20.24, 20.51, 21.19, 22.38]
        assert_almost_equal(hdr.median, median_t, decimal=2)
    except OSError:
        pytest.xfail('Multiprocess randomly crashes in Windows testing')
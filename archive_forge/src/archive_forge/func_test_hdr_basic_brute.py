import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from statsmodels.datasets import elnino
from statsmodels.graphics.functional import (
@pytest.mark.slow
@pytest.mark.matplotlib
def test_hdr_basic_brute(close_figures, reset_randomstate):
    try:
        _, hdr = hdrboxplot(data, ncomp=2, labels=labels, use_brute=True)
        assert len(hdr.extra_quantiles) == 0
        median_t = [24.247, 25.625, 25.964, 24.999, 23.648, 22.302, 21.231, 20.366, 20.168, 20.434, 21.111, 22.299]
        assert_almost_equal(hdr.median, median_t, decimal=2)
    except OSError:
        pytest.xfail('Multiprocess randomly crashes in Windows testing')
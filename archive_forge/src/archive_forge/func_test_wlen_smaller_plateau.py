import copy
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises, warns
from scipy.signal._peak_finding import (
from scipy.signal.windows import gaussian
from scipy.signal._peak_finding_utils import _local_maxima_1d, PeakPropertyWarning
@pytest.mark.filterwarnings('ignore:some peaks have a prominence of 0', 'ignore:some peaks have a width of 0')
def test_wlen_smaller_plateau(self):
    """
        Test behavior of prominence and width calculation if the given window
        length is smaller than a peak's plateau size.

        Regression test for gh-9110.
        """
    peaks, props = find_peaks([0, 1, 1, 1, 0], prominence=(None, None), width=(None, None), wlen=2)
    assert_equal(peaks, 2)
    assert_equal(props['prominences'], 0)
    assert_equal(props['widths'], 0)
    assert_equal(props['width_heights'], 1)
    for key in ('left_bases', 'right_bases', 'left_ips', 'right_ips'):
        assert_equal(props[key], peaks)
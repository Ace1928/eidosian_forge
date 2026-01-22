import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_, assert_equal
from statsmodels.stats import moment_helpers
from statsmodels.stats.moment_helpers import (cov2corr, mvsk2mc, mc2mvsk,
@pytest.mark.parametrize('mom', ms)
def test_moment_conversion(mom):
    assert_equal(mnc2cum(mc2mnc(mom[0])), mom[1])
    assert_equal(mnc2cum(mom[0]), mom[2])
    if len(mom) <= 4:
        assert_equal(mc2cum(mom[0]), mom[1])
    assert_equal(cum2mc(mom[1]), mom[0])
    assert_equal(mc2mnc(cum2mc(mom[2])), mom[0])
    if len(mom) <= 4:
        assert_equal(cum2mc(mom[1]), mom[0])
    assert_equal(cum2mc(mnc2cum(mom[0])), mnc2mc(mom[0]))
    assert_equal(mc2mnc(mnc2mc(mom[0])), mom[0])
    if len(mom[0]) == 4:
        assert_equal(mvsk2mc(mc2mvsk(mom[0])), mom[0])
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy import stats
from scipy.stats import _survival
def test_right_censored_ci(self):
    times, died = (self.t4, self.d4)
    sample = stats.CensoredData.right_censored(times, np.logical_not(died))
    res = stats.ecdf(sample)
    ref_allowance = [0.096, 0.096, 0.135, 0.162, 0.162, 0.162, 0.162, 0.162, 0.162, 0.162, 0.214, 0.246, 0.246, 0.246, 0.246, 0.341, 0.341]
    sf_ci = res.sf.confidence_interval()
    cdf_ci = res.cdf.confidence_interval()
    allowance = res.sf.probabilities - sf_ci.low.probabilities
    assert_allclose(allowance, ref_allowance, atol=0.001)
    assert_allclose(sf_ci.low.probabilities, np.clip(res.sf.probabilities - allowance, 0, 1))
    assert_allclose(sf_ci.high.probabilities, np.clip(res.sf.probabilities + allowance, 0, 1))
    assert_allclose(cdf_ci.low.probabilities, np.clip(res.cdf.probabilities - allowance, 0, 1))
    assert_allclose(cdf_ci.high.probabilities, np.clip(res.cdf.probabilities + allowance, 0, 1))
    ref_low = [0.694743, 0.694743, 0.647529, 0.591142, 0.591142, 0.591142, 0.591142, 0.591142, 0.591142, 0.591142, 0.464605, 0.370359, 0.370359, 0.370359, 0.370359, 0.160489, 0.160489]
    ref_high = [0.992802, 0.992802, 0.973299, 0.947073, 0.947073, 0.947073, 0.947073, 0.947073, 0.947073, 0.947073, 0.906422, 0.856521, 0.856521, 0.856521, 0.856521, 0.776724, 0.776724]
    sf_ci = res.sf.confidence_interval(method='log-log')
    assert_allclose(sf_ci.low.probabilities, ref_low, atol=1e-06)
    assert_allclose(sf_ci.high.probabilities, ref_high, atol=1e-06)
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.stats.contingency import relative_risk
def test_relative_risk_ci_edge_cases_10():
    result = relative_risk(exposed_cases=1, exposed_total=12, control_cases=0, control_total=30)
    assert_equal(result.relative_risk, np.inf)
    ci = result.confidence_interval()
    assert_equal((ci.low, ci.high), (np.nan, np.inf))
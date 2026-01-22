import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.stats.contingency import relative_risk
def test_relative_risk_ci_conflevel0():
    result = relative_risk(exposed_cases=4, exposed_total=12, control_cases=5, control_total=30)
    rr = result.relative_risk
    assert_allclose(rr, 2.0, rtol=1e-14)
    ci = result.confidence_interval(0)
    assert_allclose((ci.low, ci.high), (2.0, 2.0), rtol=1e-12)
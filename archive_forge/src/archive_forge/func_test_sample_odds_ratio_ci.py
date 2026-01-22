import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from .._discrete_distns import nchypergeom_fisher, hypergeom
from scipy.stats._odds_ratio import odds_ratio
from .data.fisher_exact_results_from_r import data
@pytest.mark.parametrize('case', [[0.95, 'two-sided', 0.4879913, 2.635883], [0.9, 'two-sided', 0.5588516, 2.301663]])
def test_sample_odds_ratio_ci(self, case):
    confidence_level, alternative, ref_low, ref_high = case
    table = [[10, 20], [41, 93]]
    result = odds_ratio(table, kind='sample')
    assert_allclose(result.statistic, 1.134146, rtol=1e-06)
    ci = result.confidence_interval(confidence_level, alternative)
    assert_allclose([ci.low, ci.high], [ref_low, ref_high], rtol=1e-06)
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from .._discrete_distns import nchypergeom_fisher, hypergeom
from scipy.stats._odds_ratio import odds_ratio
from .data.fisher_exact_results_from_r import data
@pytest.mark.parametrize('alternative', ['less', 'greater', 'two-sided'])
def test_sample_odds_ratio_one_sided_ci(self, alternative):
    table = [[1000, 2000], [4100, 9300]]
    res = odds_ratio(table, kind='sample')
    ref = odds_ratio(table, kind='conditional')
    assert_allclose(res.statistic, ref.statistic, atol=1e-05)
    assert_allclose(res.confidence_interval(alternative=alternative), ref.confidence_interval(alternative=alternative), atol=0.002)
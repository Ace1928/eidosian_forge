import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.stats.proportion as smprop
from statsmodels.stats.proportion import (
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.stats.tests.results.results_proportion import res_binom, res_binom_methods
def test_number_pairs_1493(self):
    ppt = smprop.proportions_chisquare_allpairs(self.n_success[:3], self.nobs[:3], multitest_method=None)
    assert_equal(len(ppt.pvals_raw), 3)
    idx = [0, 1, 3]
    assert_almost_equal(ppt.pvals_raw, self.res_ppt_pvals_raw[idx])
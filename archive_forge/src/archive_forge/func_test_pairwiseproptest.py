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
def test_pairwiseproptest(self):
    ppt = smprop.proportions_chisquare_allpairs(self.n_success, self.nobs, multitest_method=None)
    assert_almost_equal(ppt.pvals_raw, self.res_ppt_pvals_raw)
    ppt = smprop.proportions_chisquare_allpairs(self.n_success, self.nobs, multitest_method='h')
    assert_almost_equal(ppt.pval_corrected(), self.res_ppt_pvals_holm)
    pptd = smprop.proportions_chisquare_pairscontrol(self.n_success, self.nobs, multitest_method='hommel')
    assert_almost_equal(pptd.pvals_raw, ppt.pvals_raw[:len(self.nobs) - 1], decimal=13)
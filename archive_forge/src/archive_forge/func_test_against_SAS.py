import warnings
import sys
from functools import partial
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import re
from scipy import optimize, stats, special
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
from scipy.stats._binomtest import _binary_search_for_binom_tst
from scipy.stats._distr_params import distcont
@pytest.mark.parametrize('x,y,alternative,stat_expect,p_expect', mood_cases_with_ties())
def test_against_SAS(self, x, y, alternative, stat_expect, p_expect):
    """
        Example code used to generate SAS output:
        DATA myData;
        INPUT X Y;
        CARDS;
        1 0
        1 1
        1 2
        1 3
        1 4
        2 0
        2 1
        2 4
        2 9
        2 16
        ods graphics on;
        proc npar1way mood data=myData ;
           class X;
            ods output  MoodTest=mt;
        proc contents data=mt;
        proc print data=mt;
          format     Prob1 17.16 Prob2 17.16 Statistic 17.16 Z 17.16 ;
            title "Mood Two-Sample Test";
        proc print data=myData;
            title "Data for above results";
          run;
        """
    statistic, pvalue = stats.mood(x, y, alternative=alternative)
    assert_allclose(stat_expect, statistic, atol=1e-16)
    assert_allclose(p_expect, pvalue, atol=1e-16)
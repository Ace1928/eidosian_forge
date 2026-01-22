import warnings
import re
import sys
import pickle
from pathlib import Path
import os
import json
import platform
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import numpy
import numpy as np
from numpy import typecodes, array
from numpy.lib.recfunctions import rec_append_fields
from scipy import special
from scipy._lib._util import check_random_state
from scipy.integrate import (IntegrationWarning, quad, trapezoid,
import scipy.stats as stats
from scipy.stats._distn_infrastructure import argsreduce
import scipy.stats.distributions
from scipy.special import xlogy, polygamma, entr
from scipy.stats._distr_params import distcont, invdistcont
from .test_discrete_basic import distdiscrete, invdistdiscrete
from scipy.stats._continuous_distns import FitDataError, _argus_phi
from scipy.optimize import root, fmin, differential_evolution
from itertools import product
def test_gh_9403_nontail_values(self):
    for low, high in [[3, 4], [-4, -3]]:
        xvals = np.array([-np.inf, low, high, np.inf])
        xmid = (high + low) / 2.0
        cdfs = stats.truncnorm.cdf(xvals, low, high)
        sfs = stats.truncnorm.sf(xvals, low, high)
        pdfs = stats.truncnorm.pdf(xvals, low, high)
        expected_cdfs = np.array([0, 0, 1, 1])
        expected_sfs = np.array([1.0, 1.0, 0.0, 0.0])
        expected_pdfs = np.array([0, 3.3619772, 0.1015229, 0])
        if low < 0:
            expected_pdfs = np.array([0, 0.1015229, 3.3619772, 0])
        assert_almost_equal(cdfs, expected_cdfs)
        assert_almost_equal(sfs, expected_sfs)
        assert_almost_equal(pdfs, expected_pdfs)
        assert_almost_equal(np.log(expected_pdfs[1] / expected_pdfs[2]), low + 0.5)
        pvals = np.array([0, 0.5, 1.0])
        ppfs = stats.truncnorm.ppf(pvals, low, high)
        expected_ppfs = np.array([low, np.sign(low) * 3.1984741, high])
        assert_almost_equal(ppfs, expected_ppfs)
        if low < 0:
            assert_almost_equal(stats.truncnorm.sf(xmid, low, high), 0.8475544278436675)
            assert_almost_equal(stats.truncnorm.cdf(xmid, low, high), 0.1524455721563326)
        else:
            assert_almost_equal(stats.truncnorm.cdf(xmid, low, high), 0.8475544278436675)
            assert_almost_equal(stats.truncnorm.sf(xmid, low, high), 0.1524455721563326)
        pdf = stats.truncnorm.pdf(xmid, low, high)
        assert_almost_equal(np.log(pdf / expected_pdfs[2]), (xmid + 0.25) / 2)
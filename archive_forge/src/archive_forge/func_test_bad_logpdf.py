import threading
import pickle
import pytest
from copy import deepcopy
import platform
import sys
import math
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy.stats.sampling import (
from pytest import raises as assert_raises
from scipy import stats
from scipy import special
from scipy.stats import chisquare, cramervonmises
from scipy.stats._distr_params import distdiscrete, distcont
from scipy._lib._util import check_random_state
@pytest.mark.parametrize('logpdf, err, msg', bad_logpdfs_common)
def test_bad_logpdf(self, logpdf, err, msg):

    class dist:
        pass
    dist.logpdf = logpdf
    with pytest.raises(err, match=msg):
        NumericalInversePolynomial(dist, domain=[0, 5])
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
def test_logpdf_pdf_consistency(self):

    class MyDist:
        pass
    dist_pdf = MyDist()
    dist_pdf.pdf = lambda x: math.exp(-x * x / 2)
    rng1 = NumericalInversePolynomial(dist_pdf)
    dist_logpdf = MyDist()
    dist_logpdf.logpdf = lambda x: -x * x / 2
    rng2 = NumericalInversePolynomial(dist_logpdf)
    q = np.linspace(1e-05, 1 - 1e-05, num=100)
    assert_allclose(rng1.ppf(q), rng2.ppf(q))
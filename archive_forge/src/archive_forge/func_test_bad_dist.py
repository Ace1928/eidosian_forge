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
def test_bad_dist(self):

    class dist:
        ...
    msg = '`pdf` required but not found.'
    with pytest.raises(ValueError, match=msg):
        TransformedDensityRejection(dist)

    class dist:
        pdf = lambda x: 1 - x * x
    msg = '`dpdf` required but not found.'
    with pytest.raises(ValueError, match=msg):
        TransformedDensityRejection(dist)
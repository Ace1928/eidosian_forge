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
@pytest.mark.parametrize('c', [-1.0, np.nan, np.inf, 0.1, 1.0])
def test_bad_c(self, c):
    msg = '`c` must either be -0.5 or 0.'
    with pytest.raises(ValueError, match=msg):
        TransformedDensityRejection(StandardNormal(), c=-1.0)
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_raises,
import numpy.testing as npt
from scipy.special import gamma, factorial, factorial2
import scipy.stats as stats
from statsmodels.distributions.edgeworth import (_faa_di_bruno_partitions,
def test_pdf_no_roots(self):
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        ne = ExpandedNormal([0, 1])
        ne = ExpandedNormal([0, 1, 0.1, 0.1])
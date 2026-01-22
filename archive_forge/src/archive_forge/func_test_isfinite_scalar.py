import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_isfinite_scalar(self):
    assert_(not np.isfinite(np.datetime64('NaT', 'ms')))
    assert_(not np.isfinite(np.datetime64('NaT', 'ns')))
    assert_(np.isfinite(np.datetime64('2038-01-19T03:14:07')))
    assert_(not np.isfinite(np.timedelta64('NaT', 'ms')))
    assert_(np.isfinite(np.timedelta64(34, 'ms')))
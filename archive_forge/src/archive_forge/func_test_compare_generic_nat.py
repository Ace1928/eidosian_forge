import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_compare_generic_nat(self):
    assert_(np.datetime64('NaT') != np.datetime64('2000') + np.timedelta64('NaT'))
    assert_(np.datetime64('NaT') != np.datetime64('NaT', 'us'))
    assert_(np.datetime64('NaT', 'us') != np.datetime64('NaT'))
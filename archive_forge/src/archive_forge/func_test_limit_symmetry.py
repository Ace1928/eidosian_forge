import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.parametrize('time_unit', ['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as', '10D', '2M'])
def test_limit_symmetry(self, time_unit):
    """
        Dates should have symmetric limits around the unix epoch at +/-np.int64
        """
    epoch = np.datetime64(0, time_unit)
    latest = np.datetime64(np.iinfo(np.int64).max, time_unit)
    earliest = np.datetime64(-np.iinfo(np.int64).max, time_unit)
    assert earliest < epoch < latest
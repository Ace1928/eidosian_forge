import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.parametrize('time_unit', ['Y', 'M', pytest.param('W', marks=pytest.mark.xfail(reason='gh-13197')), 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as', pytest.param('10D', marks=pytest.mark.xfail(reason='similar to gh-13197'))])
@pytest.mark.parametrize('sign', [-1, 1])
def test_limit_str_roundtrip(self, time_unit, sign):
    """
        Limits should roundtrip when converted to strings.

        This tests the conversion to and from npy_datetimestruct.
        """
    limit = np.datetime64(np.iinfo(np.int64).max * sign, time_unit)
    limit_via_str = np.datetime64(str(limit), time_unit)
    assert limit_via_str == limit
import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.parametrize('val1, val2', [(np.timedelta64(7, 'Y'), np.timedelta64(3, 's')), (np.timedelta64(7, 'M'), np.timedelta64(1, 'D'))])
def test_timedelta_floor_div_error(self, val1, val2):
    with assert_raises_regex(TypeError, 'common metadata divisor'):
        val1 // val2
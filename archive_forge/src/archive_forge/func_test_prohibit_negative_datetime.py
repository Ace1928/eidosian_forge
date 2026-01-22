import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns', 'ps', 'fs', 'as'])
def test_prohibit_negative_datetime(self, unit):
    with assert_raises(TypeError):
        np.array([1], dtype=f'M8[-1{unit}]')
import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_divisor_conversion_as(self):
    assert_raises(ValueError, lambda: np.dtype('M8[as/10]'))
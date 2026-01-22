import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_0D_array(self):
    a = np.array(1)
    assert_raises(ValueError, dsplit, a, 2)
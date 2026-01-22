import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_report_max_relative_error(self):
    a = np.array([0, 1])
    b = np.array([0, 2])
    with pytest.raises(AssertionError) as exc_info:
        assert_allclose(a, b)
    msg = str(exc_info.value)
    assert_('Max relative difference: 0.5' in msg)
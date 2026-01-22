import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_field_growing_cases():
    res = np.loadtxt([''], delimiter=',', dtype=bytes)
    assert len(res) == 0
    for i in range(1, 1024):
        res = np.loadtxt([',' * i], delimiter=',', dtype=bytes)
        assert len(res) == i + 1
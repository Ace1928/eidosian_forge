import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_structure_format_int(self):
    struct_int = np.array([([1, -1],), ([123, 1],)], dtype=[('B', 'i4', 2)])
    assert_equal(np.array2string(struct_int), '[([  1,  -1],) ([123,   1],)]')
    struct_2dint = np.array([([[0, 1], [2, 3]],), ([[12, 0], [0, 0]],)], dtype=[('B', 'i4', (2, 2))])
    assert_equal(np.array2string(struct_2dint), '[([[ 0,  1], [ 2,  3]],) ([[12,  0], [ 0,  0]],)]')
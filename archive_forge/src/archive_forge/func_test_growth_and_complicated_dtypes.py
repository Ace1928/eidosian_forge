import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
@pytest.mark.parametrize(['dtype', 'data'], [('d', [1, 2, 3, 4, 5, 6, 7, 8, 9]), ('O', [1, 2, 3, 4, 5, 6, 7, 8, 9]), ('i,O', [(1, 2), (5, 4), (2, 3), (9, 8), (6, 7)]), ('2i', [(1, 2), (5, 4), (2, 3), (9, 8), (6, 7)]), (np.dtype(('O', (2, 3))), [((1, 2, 3), (3, 4, 5)), ((3, 2, 1), (5, 4, 3))])])
@pytest.mark.parametrize('length_hint', [0, 1])
def test_growth_and_complicated_dtypes(self, dtype, data, length_hint):
    dtype = np.dtype(dtype)
    data = data * 100

    class MyIter:

        def __length_hint__(self):
            return length_hint

        def __iter__(self):
            return iter(data)
    res = np.fromiter(MyIter(), dtype=dtype)
    expected = np.array(data, dtype=dtype)
    assert_array_equal(res, expected)
import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
def test_int_from_long(self):
    l = [1000000.0, 1000000000000.0, 1e+18, -1000000.0, -1000000000000.0, -1e+18]
    li = [10 ** 6, 10 ** 12, 10 ** 18, -10 ** 6, -10 ** 12, -10 ** 18]
    for T in [None, np.float64, np.int64]:
        a = np.array(l, dtype=T)
        assert_equal([int(_m) for _m in a], li)
    a = np.array(l[:3], dtype=np.uint64)
    assert_equal([int(_m) for _m in a], li[:3])
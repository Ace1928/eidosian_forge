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
def test_int_raise_behaviour(self):

    def overflow_error_func(dtype):
        dtype(np.iinfo(dtype).max + 1)
    for code in [np.int_, np.uint, np.longlong, np.ulonglong]:
        assert_raises(OverflowError, overflow_error_func, code)
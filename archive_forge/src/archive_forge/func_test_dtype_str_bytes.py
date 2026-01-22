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
@pytest.mark.parametrize('likefunc', [np.empty_like, np.full_like, np.zeros_like, np.ones_like])
@pytest.mark.parametrize('dtype', [str, bytes])
def test_dtype_str_bytes(self, likefunc, dtype):
    a = np.arange(16).reshape(2, 8)
    b = a[:, ::2]
    kwargs = {'fill_value': ''} if likefunc == np.full_like else {}
    result = likefunc(b, dtype=dtype, **kwargs)
    if dtype == str:
        assert result.strides == (16, 4)
    else:
        assert result.strides == (4, 1)
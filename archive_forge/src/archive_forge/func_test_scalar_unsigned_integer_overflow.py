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
@pytest.mark.parametrize('dtype', np.typecodes['UnsignedInteger'])
def test_scalar_unsigned_integer_overflow(dtype):
    val = np.dtype(dtype).type(8)
    with pytest.warns(RuntimeWarning, match='overflow encountered'):
        -val
    zero = np.dtype(dtype).type(0)
    -zero
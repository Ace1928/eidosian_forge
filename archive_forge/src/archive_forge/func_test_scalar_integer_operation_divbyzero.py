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
@pytest.mark.parametrize('dtype', np.typecodes['AllInteger'])
@pytest.mark.parametrize('operation', [lambda val, zero: val // zero, lambda val, zero: val % zero], ids=['//', '%'])
def test_scalar_integer_operation_divbyzero(dtype, operation):
    st = np.dtype(dtype).type
    val = st(100)
    zero = st(0)
    with pytest.warns(RuntimeWarning, match='divide by zero'):
        operation(val, zero)
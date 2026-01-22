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
@pytest.mark.parametrize('op', reasonable_operators_for_scalars)
@pytest.mark.parametrize('val', [None, 2 ** 64])
def test_clongdouble_inf_loop(op, val):
    try:
        op(np.clongdouble(3), val)
    except TypeError:
        pass
    try:
        op(val, np.longdouble(3))
    except TypeError:
        pass
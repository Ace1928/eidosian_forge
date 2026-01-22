import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
@pytest.mark.skipif(sys.flags.optimize == 2, reason='Python running -OO')
@pytest.mark.skipif(IS_PYPY, reason='PyPy does not modify tp_doc')
def test_add_same_docstring(self):
    np.add_docstring(np.ndarray.flat, np.ndarray.flat.__doc__)

    def func():
        """docstring"""
        return
    np.add_docstring(func, func.__doc__)
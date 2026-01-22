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
def test_nonzero_invalid_object(self):
    a = np.array([np.array([1, 2]), 3], dtype=object)
    assert_raises(ValueError, np.nonzero, a)

    class BoolErrors:

        def __bool__(self):
            raise ValueError('Not allowed')
    assert_raises(ValueError, np.nonzero, np.array([BoolErrors()]))
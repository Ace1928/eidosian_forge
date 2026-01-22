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
def test_move_to_end(self):
    x = np.random.randn(5, 6, 7)
    for source, expected in [(0, (6, 7, 5)), (1, (5, 7, 6)), (2, (5, 6, 7)), (-1, (5, 6, 7))]:
        actual = np.moveaxis(x, source, -1).shape
        assert_(actual, expected)
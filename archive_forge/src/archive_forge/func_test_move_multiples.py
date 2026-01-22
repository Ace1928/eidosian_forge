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
def test_move_multiples(self):
    x = np.zeros((0, 1, 2, 3))
    for source, destination, expected in [([0, 1], [2, 3], (2, 3, 0, 1)), ([2, 3], [0, 1], (2, 3, 0, 1)), ([0, 1, 2], [2, 3, 0], (2, 3, 0, 1)), ([3, 0], [1, 0], (0, 3, 1, 2)), ([0, 3], [0, 1], (0, 3, 1, 2))]:
        actual = np.moveaxis(x, source, destination).shape
        assert_(actual, expected)
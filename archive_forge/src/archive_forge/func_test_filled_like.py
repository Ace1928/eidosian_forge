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
def test_filled_like(self):
    self.check_like_function(np.full_like, 0, True)
    self.check_like_function(np.full_like, 1, True)
    self.check_like_function(np.full_like, 1000, True)
    self.check_like_function(np.full_like, 123.456, True)
    with np.errstate(invalid='ignore'):
        self.check_like_function(np.full_like, np.inf, True)
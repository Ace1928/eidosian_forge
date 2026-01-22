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
def test_C_and_F_simul(self):
    a = self.generate_all_false('f8')
    assert_raises(ValueError, np.require, a, None, ['C', 'F'])
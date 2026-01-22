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
def test_can_cast_structured_to_simple(self):
    assert_(not np.can_cast([('f1', 'i4')], 'i4'))
    assert_(np.can_cast([('f1', 'i4')], 'i4', casting='unsafe'))
    assert_(np.can_cast([('f1', 'i4')], 'i2', casting='unsafe'))
    assert_(not np.can_cast('i4,i4', 'i4', casting='unsafe'))
    assert_(not np.can_cast([('f1', [('x', 'i4')])], 'i4'))
    assert_(np.can_cast([('f1', [('x', 'i4')])], 'i4', casting='unsafe'))
    assert_(not np.can_cast([('f0', '(3,)i4')], 'i4'))
    assert_(np.can_cast([('f0', '(3,)i4')], 'i4', casting='unsafe'))
    assert_(not np.can_cast([('f0', 'i4,i4', (2,))], 'i4', casting='unsafe'))
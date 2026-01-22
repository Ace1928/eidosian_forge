from math import nan, inf
import pytest
from numpy.core import array, arange, printoptions
import numpy.polynomial as poly
from numpy.testing import assert_equal, assert_
from fractions import Fraction
from decimal import Decimal
@pytest.fixture(scope='class', autouse=True)
def use_unicode(self):
    poly.set_default_printstyle('unicode')
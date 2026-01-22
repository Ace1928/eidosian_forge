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
def test_promote_types_endian(self):
    assert_equal(np.promote_types('<i8', '<i8'), np.dtype('i8'))
    assert_equal(np.promote_types('>i8', '>i8'), np.dtype('i8'))
    assert_equal(np.promote_types('>i8', '>U16'), np.dtype('U21'))
    assert_equal(np.promote_types('<i8', '<U16'), np.dtype('U21'))
    assert_equal(np.promote_types('>U16', '>i8'), np.dtype('U21'))
    assert_equal(np.promote_types('<U16', '<i8'), np.dtype('U21'))
    assert_equal(np.promote_types('<S5', '<U8'), np.dtype('U8'))
    assert_equal(np.promote_types('>S5', '>U8'), np.dtype('U8'))
    assert_equal(np.promote_types('<U8', '<S5'), np.dtype('U8'))
    assert_equal(np.promote_types('>U8', '>S5'), np.dtype('U8'))
    assert_equal(np.promote_types('<U5', '<U8'), np.dtype('U8'))
    assert_equal(np.promote_types('>U8', '>U5'), np.dtype('U8'))
    assert_equal(np.promote_types('<M8', '<M8'), np.dtype('M8'))
    assert_equal(np.promote_types('>M8', '>M8'), np.dtype('M8'))
    assert_equal(np.promote_types('<m8', '<m8'), np.dtype('m8'))
    assert_equal(np.promote_types('>m8', '>m8'), np.dtype('m8'))
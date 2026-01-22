import fractions
import platform
import types
from typing import Any, Type
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_raises, IS_MUSL
@pytest.mark.parametrize('ftype', [np.half, np.single, np.double, np.longdouble])
def test_simple_fractions(self, ftype):
    R = fractions.Fraction
    assert_equal(R(0, 1), R(*ftype(0.0).as_integer_ratio()))
    assert_equal(R(5, 2), R(*ftype(2.5).as_integer_ratio()))
    assert_equal(R(1, 2), R(*ftype(0.5).as_integer_ratio()))
    assert_equal(R(-2100, 1), R(*ftype(-2100.0).as_integer_ratio()))
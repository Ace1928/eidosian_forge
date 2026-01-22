import fractions
import platform
import types
from typing import Any, Type
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_raises, IS_MUSL
def test_against_known_values(self):
    R = fractions.Fraction
    assert_equal(R(1075, 512), R(*np.half(2.1).as_integer_ratio()))
    assert_equal(R(-1075, 512), R(*np.half(-2.1).as_integer_ratio()))
    assert_equal(R(4404019, 2097152), R(*np.single(2.1).as_integer_ratio()))
    assert_equal(R(-4404019, 2097152), R(*np.single(-2.1).as_integer_ratio()))
    assert_equal(R(4728779608739021, 2251799813685248), R(*np.double(2.1).as_integer_ratio()))
    assert_equal(R(-4728779608739021, 2251799813685248), R(*np.double(-2.1).as_integer_ratio()))
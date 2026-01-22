import os
from os.path import join
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
import pytest
from numpy.random import (
from numpy.random._common import interface
def test_advance_symmetry(self):
    rs = Generator(self.bit_generator(*self.data1['seed']))
    state = rs.bit_generator.state
    step = -210306068529402873148182252916320501760
    rs.bit_generator.advance(step)
    val_neg = rs.integers(10)
    rs.bit_generator.state = state
    rs.bit_generator.advance(2 ** 128 + step)
    val_pos = rs.integers(10)
    rs.bit_generator.state = state
    rs.bit_generator.advance(10 * 2 ** 128 + step)
    val_big = rs.integers(10)
    assert val_neg == val_pos
    assert val_big == val_pos
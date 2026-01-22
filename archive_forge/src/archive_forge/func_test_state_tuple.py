import os
from os.path import join
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
import pytest
from numpy.random import (
from numpy.random._common import interface
def test_state_tuple(self):
    rs = Generator(self.bit_generator(*self.data1['seed']))
    bit_generator = rs.bit_generator
    state = bit_generator.state
    desired = rs.integers(2 ** 16)
    tup = (state['bit_generator'], state['state']['key'], state['state']['pos'])
    bit_generator.state = tup
    actual = rs.integers(2 ** 16)
    assert_equal(actual, desired)
    tup = tup + (0, 0.0)
    bit_generator.state = tup
    actual = rs.integers(2 ** 16)
    assert_equal(actual, desired)
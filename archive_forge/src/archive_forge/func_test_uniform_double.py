import os
from os.path import join
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
import pytest
from numpy.random import (
from numpy.random._common import interface
def test_uniform_double(self):
    rs = Generator(self.bit_generator(*self.data1['seed']))
    vals = uniform_from_uint(self.data1['data'], self.bits)
    uniforms = rs.random(len(vals))
    assert_allclose(uniforms, vals)
    assert_equal(uniforms.dtype, np.float64)
    rs = Generator(self.bit_generator(*self.data2['seed']))
    vals = uniform_from_uint(self.data2['data'], self.bits)
    uniforms = rs.random(len(vals))
    assert_allclose(uniforms, vals)
    assert_equal(uniforms.dtype, np.float64)
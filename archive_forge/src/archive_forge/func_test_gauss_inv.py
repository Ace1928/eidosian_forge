import os
from os.path import join
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
import pytest
from numpy.random import (
from numpy.random._common import interface
def test_gauss_inv(self):
    n = 25
    rs = RandomState(self.bit_generator(*self.data1['seed']))
    gauss = rs.standard_normal(n)
    assert_allclose(gauss, gauss_from_uint(self.data1['data'], n, self.bits))
    rs = RandomState(self.bit_generator(*self.data2['seed']))
    gauss = rs.standard_normal(25)
    assert_allclose(gauss, gauss_from_uint(self.data2['data'], n, self.bits))
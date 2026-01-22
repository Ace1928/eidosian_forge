import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.io._harwell_boeing import (
def test_exp_exp(self):
    self._test_equal('(E8.3E3)', ExpFormat(8, 3, 3))
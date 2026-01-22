import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.io._harwell_boeing import (
def test_simple_exp(self):
    self._test_equal('(E4.3)', ExpFormat(4, 3))
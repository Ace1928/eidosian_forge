import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.io._harwell_boeing import (
def test_simple_repeated_int(self):
    self._test_equal('(3I4)', IntFormat(4, repeat=3))
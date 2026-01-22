import os
from os.path import join
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
import pytest
from numpy.random import (
from numpy.random._common import interface
def test_invalid_init_values(self):
    bit_generator = self.bit_generator
    for st in self.invalid_init_values:
        with pytest.raises((ValueError, OverflowError)):
            bit_generator(*st)
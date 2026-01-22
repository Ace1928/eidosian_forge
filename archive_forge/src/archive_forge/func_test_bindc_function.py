from . import util
import numpy as np
import pytest
from numpy.testing import assert_allclose
def test_bindc_function(self):
    out = self.module.coddity.wat(1, 20)
    exp_out = 8
    assert out == exp_out
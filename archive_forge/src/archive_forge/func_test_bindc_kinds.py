from . import util
import numpy as np
import pytest
from numpy.testing import assert_allclose
def test_bindc_kinds(self):
    out = self.module.coddity.c_add_int64(1, 20)
    exp_out = 21
    assert out == exp_out
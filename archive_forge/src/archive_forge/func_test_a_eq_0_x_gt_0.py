import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import scipy.special as sc
from scipy.special._testutils import FuncData
def test_a_eq_0_x_gt_0(self):
    assert sc.gammaincc(0, 1) == 0
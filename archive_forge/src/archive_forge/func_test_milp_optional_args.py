import re
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from .test_linprog import magic_square
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy import sparse
def test_milp_optional_args():
    res = milp(1)
    assert res.fun == 0
    assert_array_equal(res.x, [0])
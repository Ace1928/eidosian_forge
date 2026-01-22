import operator as op
from numbers import Number
import pytest
import numpy as np
from numpy.polynomial import (
from numpy.testing import (
from numpy.polynomial.polyutils import RankWarning
def test_ufunc_override(Poly):
    p = Poly([1, 2, 3])
    x = np.ones(3)
    assert_raises(TypeError, np.add, p, x)
    assert_raises(TypeError, np.add, x, p)
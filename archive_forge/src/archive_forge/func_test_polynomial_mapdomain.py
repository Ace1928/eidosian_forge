import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
def test_polynomial_mapdomain():
    dom1 = [0, 4]
    dom2 = [1, 3]
    x = np.matrix([dom1, dom1])
    res = np.polynomial.polyutils.mapdomain(x, dom1, dom2)
    assert_(isinstance(res, np.matrix))
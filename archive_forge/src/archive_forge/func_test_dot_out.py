import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_dot_out(self):
    a = array(np.eye(3))
    out = array(np.zeros((3, 3)))
    res = dot(a, a, out=out)
    assert_(res is out)
    assert_equal(a, res)
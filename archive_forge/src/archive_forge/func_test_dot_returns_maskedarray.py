import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_dot_returns_maskedarray(self):
    a = np.eye(3)
    b = array(a)
    assert_(type(dot(a, a)) is MaskedArray)
    assert_(type(dot(a, b)) is MaskedArray)
    assert_(type(dot(b, a)) is MaskedArray)
    assert_(type(dot(b, b)) is MaskedArray)
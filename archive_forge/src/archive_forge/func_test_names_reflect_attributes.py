import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
@pytest.mark.parametrize('t', numeric_types)
def test_names_reflect_attributes(self, t):
    """ Test that names correspond to where the type is under ``np.`` """
    assert getattr(np, t.__name__) is t
import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
def test_inner_scalar_and_matrix():
    for dt in np.typecodes['AllInteger'] + np.typecodes['AllFloat'] + '?':
        sca = np.array(3, dtype=dt)[()]
        arr = np.matrix([[1, 2], [3, 4]], dtype=dt)
        desired = np.matrix([[3, 6], [9, 12]], dtype=dt)
        assert_equal(np.inner(arr, sca), desired)
        assert_equal(np.inner(sca, arr), desired)
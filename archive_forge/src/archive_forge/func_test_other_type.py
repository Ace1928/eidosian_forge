import sys
import itertools
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_PYPY
def test_other_type(self):
    assert_equal(np.sctype2char(float), 'd')
    assert_equal(np.sctype2char(list), 'O')
    assert_equal(np.sctype2char(np.ndarray), 'O')
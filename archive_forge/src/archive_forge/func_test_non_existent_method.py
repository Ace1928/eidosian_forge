import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
def test_non_existent_method(self):

    def fail():
        _vec_string('a', np.bytes_, 'bogus')
    assert_raises(AttributeError, fail)
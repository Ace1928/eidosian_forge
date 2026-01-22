import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
def test_non_string_array(self):

    def fail():
        _vec_string(1, np.bytes_, 'strip')
    assert_raises(TypeError, fail)
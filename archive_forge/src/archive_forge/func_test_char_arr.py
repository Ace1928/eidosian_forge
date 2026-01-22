import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
def test_char_arr(self):
    for out in (self.module.string_test.strarr, self.module.string_test.strarr77):
        expected = (5, 7)
        assert out.shape == expected
        expected = '|S12'
        assert out.dtype == expected
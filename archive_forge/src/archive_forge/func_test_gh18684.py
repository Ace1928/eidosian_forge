import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
def test_gh18684(self):
    f = getattr(self.module, self.fprefix + '_gh18684')
    x = np.array(['abcde', 'fghij'], dtype='S5')
    y = f(x)
    assert_array_equal(x, y)
import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
def test_gh24662(self):
    self.module.string_inout_optional()
    a = np.array('hi', dtype='S32')
    self.module.string_inout_optional(a)
    assert 'output string' in a.tobytes().decode()
    with pytest.raises(Exception):
        aa = 'Hi'
        self.module.string_inout_optional(aa)
import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_dtype_linewidth_wrapping(self):
    np.set_printoptions(linewidth=75)
    assert_equal(repr(np.arange(10, 20.0, dtype='f4')), 'array([10., 11., 12., 13., 14., 15., 16., 17., 18., 19.], dtype=float32)')
    assert_equal(repr(np.arange(10, 23.0, dtype='f4')), textwrap.dedent('            array([10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22.],\n                  dtype=float32)'))
    styp = '<U4'
    assert_equal(repr(np.ones(3, dtype=styp)), "array(['1', '1', '1'], dtype='{}')".format(styp))
    assert_equal(repr(np.ones(12, dtype=styp)), textwrap.dedent("            array(['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],\n                  dtype='{}')".format(styp)))
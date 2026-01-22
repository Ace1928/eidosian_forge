import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_0d_arrays(self):
    assert_equal(str(np.array('café', '<U4')), 'café')
    assert_equal(repr(np.array('café', '<U4')), "array('café', dtype='<U4')")
    assert_equal(str(np.array('test', np.str_)), 'test')
    a = np.zeros(1, dtype=[('a', '<i4', (3,))])
    assert_equal(str(a[0]), '([0, 0, 0],)')
    assert_equal(repr(np.datetime64('2005-02-25')[...]), "array('2005-02-25', dtype='datetime64[D]')")
    assert_equal(repr(np.timedelta64('10', 'Y')[...]), "array(10, dtype='timedelta64[Y]')")
    x = np.array(1)
    np.set_printoptions(formatter={'all': lambda x: 'test'})
    assert_equal(repr(x), 'array(test)')
    assert_equal(str(x), '1')
    assert_warns(DeprecationWarning, np.array2string, np.array(1.0), style=repr)
    np.array2string(np.array(1.0), style=repr, legacy='1.13')
    np.array2string(np.array(1.0), legacy='1.13')
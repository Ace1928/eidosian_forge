import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
def test_inout_varia(self):
    f = getattr(self.module, self.fprefix + '_inout')
    a = np.array('abc', dtype='S3')
    f(a, 'A')
    assert_array_equal(a, np.array('Abc', dtype=a.dtype))
    a = np.array(['abc'], dtype='S3')
    f(a, 'A')
    assert_array_equal(a, np.array(['Abc'], dtype=a.dtype))
    try:
        f('abc', 'A')
    except ValueError as msg:
        if not str(msg).endswith(' got 3-str'):
            raise
    else:
        raise SystemError(f'{f.__name__} should have failed on str value')
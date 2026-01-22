import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
def test_input_varia(self):
    f = getattr(self.module, self.fprefix + '_input')
    assert_equal(f('a'), ord('a'))
    assert_equal(f(b'a'), ord(b'a'))
    assert_equal(f(''), 0)
    assert_equal(f(b''), 0)
    assert_equal(f(b'\x00'), 0)
    assert_equal(f('ab'), ord('a'))
    assert_equal(f(b'ab'), ord('a'))
    assert_equal(f(['a']), ord('a'))
    assert_equal(f(np.array(b'a')), ord('a'))
    assert_equal(f(np.array([b'a'])), ord('a'))
    a = np.array('a')
    assert_equal(f(a), ord('a'))
    a = np.array(['a'])
    assert_equal(f(a), ord('a'))
    try:
        f([])
    except IndexError as msg:
        if not str(msg).endswith(' got 0-list'):
            raise
    else:
        raise SystemError(f'{f.__name__} should have failed on empty list')
    try:
        f(97)
    except TypeError as msg:
        if not str(msg).endswith(' got int instance'):
            raise
    else:
        raise SystemError(f'{f.__name__} should have failed on int value')
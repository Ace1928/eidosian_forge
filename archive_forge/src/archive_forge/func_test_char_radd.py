import pytest
import numpy as np
from numpy.testing import assert_, assert_raises
def test_char_radd(self):
    np_s = np.bytes_('abc')
    np_u = np.str_('abc')
    s = b'def'
    u = 'def'
    assert_(np_s.__radd__(np_s) is NotImplemented)
    assert_(np_s.__radd__(np_u) is NotImplemented)
    assert_(np_s.__radd__(s) is NotImplemented)
    assert_(np_s.__radd__(u) is NotImplemented)
    assert_(np_u.__radd__(np_s) is NotImplemented)
    assert_(np_u.__radd__(np_u) is NotImplemented)
    assert_(np_u.__radd__(s) is NotImplemented)
    assert_(np_u.__radd__(u) is NotImplemented)
    assert_(s + np_s == b'defabc')
    assert_(u + np_u == 'defabc')

    class MyStr(str, np.generic):
        pass
    with assert_raises(TypeError):
        ret = s + MyStr('abc')

    class MyBytes(bytes, np.generic):
        pass
    ret = s + MyBytes(b'abc')
    assert type(ret) is type(s)
    assert ret == b'defabc'
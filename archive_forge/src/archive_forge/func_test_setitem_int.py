import array
import pytest
import rpy2.rinterface as ri
def test_setitem_int():
    vec = ri.ByteSexpVector((b'a', b'b', b'c'))
    vec[1] = ord(b'z')
    assert vec[1] == ord(b'z')
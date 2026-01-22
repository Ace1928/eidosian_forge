import array
import pytest
import rpy2.rinterface as ri
def test_getitem_slice():
    vec = ri.ByteSexpVector((b'a', b'b', b'c'))
    assert tuple(vec[:2]) == (ord(b'a'), ord(b'b'))
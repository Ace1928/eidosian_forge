import array
import pytest
import rpy2.rinterface as ri
def test_from_memoryview():
    a = array.array('b', b'abcdefg')
    mv = memoryview(a)
    vec = ri.ByteSexpVector.from_memoryview(mv)
    assert tuple(b'abcdefg') == tuple(vec)
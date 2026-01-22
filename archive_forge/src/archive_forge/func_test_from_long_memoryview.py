import array
import pytest
import struct
import sys
import rpy2.rinterface as ri
def test_from_long_memoryview():
    arrtype = 'l'
    if ri._rinterface.ffi.sizeof('int') == ri._rinterface.ffi.sizeof('long'):
        arrtype = 'q'
    a = array.array(arrtype, range(3, 103))
    mv = memoryview(a)
    with pytest.raises(ValueError):
        ri.IntSexpVector.from_memoryview(mv)
import array
import pytest
import struct
import sys
import rpy2.rinterface as ri
def test_getslice():
    vec = ri.IntSexpVector([1, 2, 3])
    vec = vec[0:2]
    assert len(vec) == 2
    assert vec[0] == 1
    assert vec[1] == 2
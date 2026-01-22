import array
import pytest
import struct
import sys
import rpy2.rinterface as ri
def test_init_from_iter():
    seq = range(3)
    v = ri.IntSexpVector(seq)
    assert len(v) == 3
    for x, y in zip(seq, v):
        assert x == y
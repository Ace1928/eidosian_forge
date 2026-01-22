import array
import pytest
import rpy2.rinterface as ri
def test_init_from_bytes():
    seq = b'abc'
    v = ri.ByteSexpVector(seq)
    assert len(v) == 3
    for x, y in zip(seq, v):
        assert x == y
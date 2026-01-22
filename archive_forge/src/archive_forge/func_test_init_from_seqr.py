import pytest
import rpy2.rinterface as ri
def test_init_from_seqr():
    seq = [1 + 2j, 5 + 7j, 0 + 1j]
    v = ri.ComplexSexpVector(seq)
    assert len(v) == 3
    for x, y in zip(seq, v):
        assert x == y
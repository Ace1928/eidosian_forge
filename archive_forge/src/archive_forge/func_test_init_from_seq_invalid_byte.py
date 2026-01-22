import array
import pytest
import rpy2.rinterface as ri
def test_init_from_seq_invalid_byte():
    seq = (b'a', [], b'c')
    with pytest.raises(ValueError):
        ri.ByteSexpVector(seq)
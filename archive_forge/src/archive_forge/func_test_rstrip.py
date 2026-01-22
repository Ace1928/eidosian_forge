import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
def test_rstrip(self):
    assert_(issubclass(self.A.rstrip().dtype.type, np.bytes_))
    tgt = [[b' abc', b''], [b'12345', b'MixedCase'], [b'123 \t 345', b'UPPER']]
    assert_array_equal(self.A.rstrip(), tgt)
    tgt = [[b' abc ', b''], [b'1234', b'MixedCase'], [b'123 \t 345 \x00', b'UPP']]
    assert_array_equal(self.A.rstrip([b'5', b'ER']), tgt)
    tgt = [[' Σ', ''], ['12345', 'MixedCase'], ['123 \t 345', 'UPPER']]
    assert_(issubclass(self.B.rstrip().dtype.type, np.str_))
    assert_array_equal(self.B.rstrip(), tgt)
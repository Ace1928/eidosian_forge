import pytest
import numpy as np
from numpy.testing import (
def test_superclass(self):
    s = np.str_(b'\\x61', encoding='unicode-escape')
    assert s == 'a'
    s = np.str_(b'\\x61', 'unicode-escape')
    assert s == 'a'
    with pytest.raises(UnicodeDecodeError):
        np.str_(b'\\xx', encoding='unicode-escape')
    with pytest.raises(UnicodeDecodeError):
        np.str_(b'\\xx', 'unicode-escape')
    assert np.bytes_(-2) == b'-2'
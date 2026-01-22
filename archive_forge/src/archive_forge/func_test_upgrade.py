import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_upgrade(self):
    """Tests the upgrade method."""
    converter = StringConverter()
    assert_equal(converter._status, 0)
    assert_equal(converter.upgrade('0'), 0)
    assert_equal(converter._status, 1)
    import numpy.core.numeric as nx
    status_offset = int(nx.dtype(nx.int_).itemsize < nx.dtype(nx.int64).itemsize)
    assert_equal(converter.upgrade('17179869184'), 17179869184)
    assert_equal(converter._status, 1 + status_offset)
    assert_allclose(converter.upgrade('0.'), 0.0)
    assert_equal(converter._status, 2 + status_offset)
    assert_equal(converter.upgrade('0j'), complex('0j'))
    assert_equal(converter._status, 3 + status_offset)
    for s in ['a', b'a']:
        res = converter.upgrade(s)
        assert_(type(res) is str)
        assert_equal(res, 'a')
        assert_equal(converter._status, 8 + status_offset)
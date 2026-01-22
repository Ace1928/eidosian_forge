import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import get_buffer_info
import pytest
from numpy.testing import assert_, assert_equal, assert_raises
def test_datetime_memoryview(self):
    dt1 = np.datetime64('2016-01-01')
    dt2 = np.datetime64('2017-01-01')
    expected = dict(strides=(1,), itemsize=1, ndim=1, shape=(8,), format='B', readonly=True)
    v = memoryview(dt1)
    assert self._as_dict(v) == expected
    v = memoryview(dt2 - dt1)
    assert self._as_dict(v) == expected
    dt = np.dtype([('a', 'uint16'), ('b', 'M8[s]')])
    a = np.empty(1, dt)
    assert_raises((ValueError, BufferError), memoryview, a[0])
    with pytest.raises(BufferError, match='scalar buffer is readonly'):
        get_buffer_info(dt1, ['WRITABLE'])
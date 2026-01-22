import array
import pytest
import struct
import sys
import rpy2.rinterface as ri
@pytest.mark.skipif(not has_numpy, reason='numpy currently required for memoryview to work.')
def test_memoryview_2d():
    shape = (5, 2)
    values = tuple(range(10))
    rarray = ri.baseenv['array'](ri.IntSexpVector(values), dim=ri.IntSexpVector(shape))
    mv = rarray.memoryview()
    assert mv.f_contiguous is True
    assert mv.shape == shape
    assert mv.tolist() == [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]
    rarray[0] = 10
    assert mv.tolist() == [[10, 5], [1, 6], [2, 7], [3, 8], [4, 9]]
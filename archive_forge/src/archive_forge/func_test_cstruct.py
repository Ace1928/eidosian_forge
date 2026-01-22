import sys
import pytest
import numpy as np
from numpy.testing import extbuild
@pytest.mark.skipif(sys.version_info >= (3, 12), reason='no numpy.distutils')
@pytest.mark.slow
def test_cstruct(get_module):

    class data_source:
        """
        This class is for testing the timing of the PyCapsule destructor
        invoked when numpy release its reference to the shared data as part of
        the numpy array interface protocol. If the PyCapsule destructor is
        called early the shared data is freed and invalid memory accesses will
        occur.
        """

        def __init__(self, size, value):
            self.size = size
            self.value = value

        @property
        def __array_struct__(self):
            return get_module.new_array_struct(self.size, self.value)
    stderr = sys.__stderr__
    expected_value = -3.1415
    multiplier = -10000.0
    stderr.write(' ---- create an object to share data ---- \n')
    buf = data_source(256, expected_value)
    stderr.write(' ---- OK!\n\n')
    stderr.write(' ---- share data via the array interface protocol ---- \n')
    arr = np.array(buf, copy=False)
    stderr.write('arr.__array_interface___ = %s\n' % str(arr.__array_interface__))
    stderr.write('arr.base = %s\n' % str(arr.base))
    stderr.write(' ---- OK!\n\n')
    stderr.write(' ---- destroy the object that shared data ---- \n')
    buf = None
    stderr.write(' ---- OK!\n\n')
    assert np.allclose(arr, expected_value)
    stderr.write(' ---- read shared data ---- \n')
    stderr.write('arr = %s\n' % str(arr))
    stderr.write(' ---- OK!\n\n')
    stderr.write(' ---- modify shared data ---- \n')
    arr *= multiplier
    expected_value *= multiplier
    stderr.write('arr.__array_interface___ = %s\n' % str(arr.__array_interface__))
    stderr.write('arr.base = %s\n' % str(arr.base))
    stderr.write(' ---- OK!\n\n')
    stderr.write(' ---- read modified shared data ---- \n')
    stderr.write('arr = %s\n' % str(arr))
    stderr.write(' ---- OK!\n\n')
    assert np.allclose(arr, expected_value)
    stderr.write(' ---- free shared data ---- \n')
    arr = None
    stderr.write(' ---- OK!\n\n')
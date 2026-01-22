import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
@pytest.mark.skipif(not has_numpy, reason='package numpy cannot be imported')
@pytest.mark.parametrize('constructor', (numpy.int32, numpy.int64, numpy.uint32, numpy.uint64))
def test_scalar_int(self, constructor):
    np_value = constructor(100)
    with (robjects.default_converter + rpyn.converter).context() as cv:
        r_vec = cv.py2rpy(np_value)
    r_scalar = numpy.array(r_vec)[0]
    assert np_value == r_scalar
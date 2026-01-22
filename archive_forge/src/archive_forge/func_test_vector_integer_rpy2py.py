import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
def test_vector_integer_rpy2py(self):
    l = [1, 2, 3]
    i = rinterface.IntSexpVector(l)
    with (robjects.default_converter + rpyn.converter).context() as cv:
        converted = cv.rpy2py(i)
    assert isinstance(converted, numpy.ndarray)
    assert tuple(l) == tuple(converted)
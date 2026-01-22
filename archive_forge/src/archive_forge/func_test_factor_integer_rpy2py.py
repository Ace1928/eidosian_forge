import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
def test_factor_integer_rpy2py(self):
    l = ['a', 'b', 'a']
    f = robjects.FactorVector(l)
    with (robjects.default_converter + rpyn.converter).context() as cv:
        converted = cv.rpy2py(f)
    assert isinstance(converted, numpy.ndarray)
    assert tuple(l) == tuple(converted)
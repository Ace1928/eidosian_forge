import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
def test_assign_numpy_object(self):
    x = numpy.arange(-10.0, 10.0, 1)
    env = robjects.Environment()
    with (robjects.default_converter + rpyn.converter).context() as cv:
        env['x'] = x
    assert len(env) == 1
    with robjects.default_converter.context() as lc:
        x_r = env['x']
    assert robjects.rinterface.RTYPES.REALSXP == x_r.typeof
    assert tuple(x_r.dim) == (20,)
import pytest
import inspect
import rpy2.robjects as robjects
import array
def test_init_from_existing():
    ri_f = rinterface.baseenv.find('sum')
    ro_f = Function(ri_f)
    assert identical(ri_f, ro_f)[0] == True
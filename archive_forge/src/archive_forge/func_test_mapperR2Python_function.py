import array
import pytest
import rpy2.rinterface_lib.sexp
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import conversion
def test_mapperR2Python_function():
    sexp = rinterface.globalenv.find('plot')
    ob = robjects.default_converter.rpy2py(sexp)
    assert isinstance(ob, robjects.Function)
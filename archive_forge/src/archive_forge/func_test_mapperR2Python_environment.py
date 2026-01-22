import array
import pytest
import rpy2.rinterface_lib.sexp
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import conversion
def test_mapperR2Python_environment():
    sexp = rinterface.globalenv.find('.GlobalEnv')
    assert isinstance(robjects.default_converter.rpy2py(sexp), robjects.Environment)
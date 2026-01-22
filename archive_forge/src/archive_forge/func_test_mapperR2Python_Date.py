import array
import pytest
import rpy2.rinterface_lib.sexp
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import conversion
def test_mapperR2Python_Date():
    sexp = rinterface.baseenv.find('as.Date')('2020-01-01')
    assert isinstance(robjects.default_converter.rpy2py(sexp), robjects.vectors.DateVector)
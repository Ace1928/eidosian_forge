import array
import pytest
import rpy2.rinterface_lib.sexp
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import conversion
def test_missingconversion():
    with conversion.localconverter(conversion.missingconverter) as cv:
        with pytest.raises(NotImplementedError):
            cv.py2rpy(1)
        with pytest.raises(NotImplementedError):
            cv.rpy2py(robjects.globalenv)
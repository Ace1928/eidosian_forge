import array
import pytest
import rpy2.rinterface_lib.sexp
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import conversion
def test_mapperR2Python_s4(_set_class_AB):
    classname = rinterface.StrSexpVector(['A'])
    one = rinterface.IntSexpVector([1])
    sexp = rinterface.globalenv['A'](x=one)
    assert isinstance(robjects.default_converter.rpy2py(sexp), robjects.RS4)
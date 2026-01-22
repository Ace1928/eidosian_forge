import array
import pytest
import rpy2.rinterface_lib.sexp
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import conversion
def test_mapperPy2R_taggedlist():
    py = robjects.rlc.TaggedList(('a', 'b'), tags=('foo', 'bar'))
    robj = robjects.default_converter.py2rpy(py)
    assert isinstance(robj, robjects.Vector)
    assert len(robj) == 2
    assert tuple(robj.names) == ('foo', 'bar')
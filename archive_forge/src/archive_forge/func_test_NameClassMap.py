import array
import pytest
import rpy2.rinterface_lib.sexp
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import conversion
def test_NameClassMap():
    ncm = conversion.NameClassMap(object)
    classnames = ('A', 'B')
    assert ncm.find_key(classnames) is None
    assert ncm.find(classnames) is object
    ncm['B'] = list
    assert ncm.find_key(classnames) == 'B'
    assert ncm.find(classnames) is list
    ncm['A'] = tuple
    assert ncm.find_key(classnames) == 'A'
    assert ncm.find(classnames) is tuple
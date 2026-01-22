import pytest
from .. import utils
import rpy2.rinterface as rinterface
@pytest.mark.skip(reason='WIP')
def test_from_pyobject_protected():
    pyobject = 'ahaha'
    sexp_new = rinterface.SexpExtPtr.from_pyobject(pyobject, protected=rinterface.StrSexpVector('c'))
    assert sexp_new.typeof == rinterface.RTYPES.EXTPTRSXP
    assert sexp_new.__protected__[0] == 'c'
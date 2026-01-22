import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_find_functiononly():
    hist = rinterface.globalenv.find('hist', wantfun=False)
    assert rinterface.RTYPES.CLOSXP == hist.typeof
    rinterface.globalenv['hist'] = rinterface.StrSexpVector(['foo'])
    with pytest.raises(KeyError):
        rinterface.globalenv.find('hist', wantfun=True)
import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_find_onlyfromloadedlibrary():
    with pytest.raises(KeyError):
        rinterface.globalenv.find('survfit')
    try:
        rinterface.evalr('library("survival")')
        sfit_R = rinterface.globalenv.find('survfit')
        assert isinstance(sfit_R, rinterface.SexpClosure)
    finally:
        rinterface.evalr('detach("package:survival")')
import pytest
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
def test_typeof():
    sexp = rinterface.globalenv.find('plot')
    assert sexp.typeof == rinterface.RTYPES.CLOSXP
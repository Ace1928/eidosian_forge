import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_do_slot_missing():
    sexp = rinterface.baseenv.find('pi')
    with pytest.raises(LookupError):
        sexp.do_slot('foo')
import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_do_slot_empty_string():
    sexp = rinterface.baseenv.find('pi')
    with pytest.raises(ValueError):
        sexp.do_slot('')
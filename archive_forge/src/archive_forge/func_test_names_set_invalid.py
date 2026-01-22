import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_names_set_invalid():
    sexp = rinterface.IntSexpVector([1, 2, 3])
    assert sexp.names.rid == rinterface.NULL.rid
    with pytest.raises(ValueError):
        sexp.names = ('a', 'b', 'c')
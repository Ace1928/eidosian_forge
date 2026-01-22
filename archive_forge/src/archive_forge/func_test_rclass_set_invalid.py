import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_rclass_set_invalid():
    sexp = rinterface.IntSexpVector([1, 2, 3])
    with pytest.raises(TypeError):
        sexp.rclass = rinterface.StrSexpVector(123)
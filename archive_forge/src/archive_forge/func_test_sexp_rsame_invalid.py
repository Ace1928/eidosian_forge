import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_sexp_rsame_invalid():
    sexp_a = rinterface.baseenv.find('letters')
    with pytest.raises(ValueError):
        sexp_a.rsame('foo')
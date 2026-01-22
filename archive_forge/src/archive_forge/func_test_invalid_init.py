import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_invalid_init():
    with pytest.raises(ValueError):
        rinterface.Sexp('a')
import copy
import gc
import pytest
import rpy2.rinterface as rinterface
def test_sexp_rsame_true():
    sexp_a = rinterface.baseenv.find('letters')
    sexp_b = rinterface.baseenv.find('letters')
    assert sexp_a.rsame(sexp_b)
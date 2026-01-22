import copy
import gc
import pytest
import rpy2.rinterface as rinterface
@pytest.mark.xfail(reason='WIP')
def test_deepcopy():
    sexp = rinterface.IntSexpVector([1, 2, 3])
    assert sexp.named == 0
    rinterface.baseenv.find('identity')(sexp)
    assert sexp.named >= 2
    sexp2 = sexp.__deepcopy__()
    assert sexp.typeof == sexp2.typeof
    assert list(sexp) == list(sexp2)
    assert not sexp.rsame(sexp2)
    assert sexp2.named == 0
    sexp3 = copy.deepcopy(sexp)
    assert sexp.typeof == sexp3.typeof
    assert list(sexp) == list(sexp3)
    assert not sexp.rsame(sexp3)
    assert sexp3.named == 0
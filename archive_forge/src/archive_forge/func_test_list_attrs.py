import copy
import gc
import pytest
import rpy2.rinterface as rinterface
@pytest.mark.parametrize('cls', (rinterface.IntSexpVector, rinterface.ListSexpVector))
def test_list_attrs(cls):
    x = cls((1, 2, 3))
    assert len(x.list_attrs()) == 0
    x.do_slot_assign('a', rinterface.IntSexpVector((33,)))
    assert len(x.list_attrs()) == 1
    assert 'a' in x.list_attrs()
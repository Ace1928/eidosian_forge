import pytest
from rpy2 import rinterface
from rpy2.rinterface import memorymanagement
def test_rmemory_manager_unprotect_invalid():
    with memorymanagement.rmemory() as rmemory:
        assert rmemory.count == 0
        with pytest.raises(ValueError):
            rmemory.unprotect(2)
    assert rmemory.count == 0
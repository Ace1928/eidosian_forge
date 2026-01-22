import pytest
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
def test_new():
    x = 'a'
    with pytest.raises(ValueError):
        rinterface.SexpClosure(x)
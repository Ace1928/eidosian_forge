import pytest
import rpy2.rinterface as rinterface
def test_new_missing():
    with pytest.raises(TypeError):
        rinterface.SexpSymbol()
import pytest
import rpy2.rinterface as rinterface
def test_new_str():
    symbol = rinterface.SexpSymbol('pi')
    assert 'pi' == str(symbol)
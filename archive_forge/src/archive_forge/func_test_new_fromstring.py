import pytest
import rpy2.rinterface as rinterface
def test_new_fromstring():
    symbol = rinterface.SexpSymbol('pi')
    evalsymbol = rinterface.baseenv['eval'](symbol)
    assert evalsymbol.rid == rinterface.baseenv['pi'].rid
import pytest
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
def test_closureenv_isenv():
    exp = rinterface.parse('function() { }')
    fun = rinterface.baseenv['eval'](exp)
    assert isinstance(fun.closureenv, rinterface.SexpEnvironment)
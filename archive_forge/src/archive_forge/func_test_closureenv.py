import pytest
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
def test_closureenv():
    assert 'y' not in rinterface.globalenv
    exp = rinterface.parse('function(x) { x[y] }')
    fun = rinterface.baseenv['eval'](exp)
    vec = rinterface.baseenv['letters']
    assert isinstance(fun.closureenv, rinterface.SexpEnvironment)
    with pytest.raises(rinterface.embedded.RRuntimeError):
        with pytest.warns(rinterface.RRuntimeWarning):
            fun(vec)
    fun.closureenv['y'] = rinterface.IntSexpVector([1])
    assert 'a' == fun(vec)[0]
    fun.closureenv['y'] = rinterface.IntSexpVector([2])
    assert 'b' == fun(vec)[0]
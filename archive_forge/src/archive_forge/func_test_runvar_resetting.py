import pytest
from trio import run
from trio.lowlevel import RunVar, RunVarToken
from ... import _core
def test_runvar_resetting() -> None:
    t1 = RunVar[str]('test1')
    t2 = RunVar[str]('test2', default='dogfish')
    t3 = RunVar[str]('test3')

    async def reset_check() -> None:
        token = t1.set('moonfish')
        assert t1.get() == 'moonfish'
        t1.reset(token)
        with pytest.raises(TypeError):
            t1.reset(None)
        with pytest.raises(LookupError):
            t1.get()
        token2 = t2.set('catdogfish')
        assert t2.get() == 'catdogfish'
        t2.reset(token2)
        assert t2.get() == 'dogfish'
        with pytest.raises(ValueError, match='^token has already been used$'):
            t2.reset(token2)
        token3 = t3.set('basculin')
        assert t3.get() == 'basculin'
        with pytest.raises(ValueError, match='^token is not for us$'):
            t1.reset(token3)
    run(reset_check)
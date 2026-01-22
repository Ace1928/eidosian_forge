import pytest
from IPython.core.error import TryNext
from IPython.core.hooks import CommandChainDispatcher
def test_command_chain_dispatcher_ff():
    """Test two failing hooks"""
    fail1 = Fail('fail1')
    fail2 = Fail('fail2')
    dp = CommandChainDispatcher([(0, fail1), (10, fail2)])
    with pytest.raises(TryNext) as e:
        dp()
    assert str(e.value) == 'fail2'
    assert fail1.called is True
    assert fail2.called is True
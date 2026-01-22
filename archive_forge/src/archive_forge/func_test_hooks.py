import pytest
from requests import hooks
@pytest.mark.parametrize('hooks_list, result', ((hook, 'ata'), ([hook, lambda x: None, hook], 'ta')))
def test_hooks(hooks_list, result):
    assert hooks.dispatch_hook('response', {'response': hooks_list}, 'Data') == result
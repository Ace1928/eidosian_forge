from stevedore import hook
from stevedore.tests import utils
def test_get_by_name_missing(self):
    em = hook.HookManager('stevedore.test.extension', 't1', invoke_on_load=True, invoke_args=('a',), invoke_kwds={'b': 'B'})
    try:
        em['t2']
    except KeyError:
        pass
    else:
        assert False, 'Failed to raise KeyError'
from stevedore import hook
from stevedore.tests import utils
def test_get_by_name(self):
    em = hook.HookManager('stevedore.test.extension', 't1', invoke_on_load=True, invoke_args=('a',), invoke_kwds={'b': 'B'})
    e_list = em['t1']
    self.assertEqual(len(e_list), 1)
    e = e_list[0]
    self.assertEqual(e.name, 't1')
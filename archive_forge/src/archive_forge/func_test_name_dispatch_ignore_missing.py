from stevedore import dispatch
from stevedore.tests import utils
def test_name_dispatch_ignore_missing(self):

    def invoke(ep, *args, **kwds):
        return (ep.name, args, kwds)
    em = dispatch.NameDispatchExtensionManager('stevedore.test.extension', lambda *args, **kwds: True, invoke_on_load=True, invoke_args=('a',), invoke_kwds={'b': 'B'})
    results = em.map(['t3', 't1'], invoke, 'first', named='named value')
    expected = [('t1', ('first',), {'named': 'named value'})]
    self.assertEqual(results, expected)
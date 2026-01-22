from twisted.python import context
from twisted.trial.unittest import SynchronousTestCase
def test_setDefault(self):
    """
        A default value may be set for a key in the context using
        L{twisted.python.context.setDefault}.
        """
    key = object()
    self.addCleanup(context.defaultContextDict.pop, key, None)
    context.setDefault(key, 'y')
    self.assertEqual('y', context.get(key))
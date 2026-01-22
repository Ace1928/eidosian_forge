import pickle
from twisted.internet.protocol import Protocol, ReconnectingClientFactory
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_serializeUnused(self):
    """
        A L{ReconnectingClientFactory} which hasn't been used for anything
        can be pickled and unpickled and end up with the same state.
        """
    original = ReconnectingClientFactory()
    reconstituted = pickle.loads(pickle.dumps(original))
    self.assertEqual(original.__dict__, reconstituted.__dict__)
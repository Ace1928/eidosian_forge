import pickle
from twisted.internet.protocol import Protocol, ReconnectingClientFactory
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_parametrizedClock(self):
    """
        The clock used by L{ReconnectingClientFactory} can be parametrized, so
        that one can cleanly test reconnections.
        """
    clock = Clock()
    factory = ReconnectingClientFactory()
    factory.clock = clock
    factory.clientConnectionLost(FakeConnector(), None)
    self.assertEqual(len(clock.calls), 1)
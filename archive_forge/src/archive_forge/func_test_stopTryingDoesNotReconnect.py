import pickle
from twisted.internet.protocol import Protocol, ReconnectingClientFactory
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_stopTryingDoesNotReconnect(self):
    """
        Calling stopTrying on a L{ReconnectingClientFactory} doesn't attempt a
        retry on any active connector.
        """

    class FactoryAwareFakeConnector(FakeConnector):
        attemptedRetry = False

        def stopConnecting(self):
            """
                Behave as though an ongoing connection attempt has now
                failed, and notify the factory of this.
                """
            f.clientConnectionFailed(self, None)

        def connect(self):
            """
                Record an attempt to reconnect, since this is what we
                are trying to avoid.
                """
            self.attemptedRetry = True
    f = ReconnectingClientFactory()
    f.clock = Clock()
    f.connector = FactoryAwareFakeConnector()
    f.stopTrying()
    self.assertFalse(f.connector.attemptedRetry)
    self.assertFalse(f.clock.getDelayedCalls())
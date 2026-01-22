from twisted.internet import error
from twisted.trial.unittest import SynchronousTestCase
def test_strArgs(self) -> None:
    """
        Any arguments passed to L{ConnectionAborted} are included in its
        message.
        """
    self.assertEqual('Connection was aborted locally using ITCPTransport.abortConnection: foo bar.', str(error.ConnectionAborted('foo', 'bar')))
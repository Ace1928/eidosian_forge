from twisted.internet.interfaces import IReactorThreads, IReactorTime
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.log import msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
def test_delayedCallStopsReactor(self):
    """
        The reactor can be stopped by a delayed call.
        """
    reactor = self.buildReactor()
    reactor.callLater(0, reactor.stop)
    reactor.run()
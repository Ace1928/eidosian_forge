from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
def getDeferred(self):
    """
        A sample function that returns a L{Deferred} that can be fired on
        demand, by L{CancellationTests.deferredGotten}.

        @return: L{Deferred} that can be fired on demand.
        """
    self.deferredsOutstanding.append(Deferred())
    return self.deferredsOutstanding[-1]
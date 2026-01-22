import pickle
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.application import internet
from twisted.application.internet import (
from twisted.internet import task
from twisted.internet.defer import CancelledError, Deferred
from twisted.internet.interfaces import (
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.logger import formatEvent, globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_failedCallLogsError(self):
    """
        When function passed to L{TimerService} returns a deferred that errbacks,
        the exception is logged, and L{TimerService.stopService} doesn't raise an error.
        """
    self.timer.startService()
    self.deferred.errback(Failure(ZeroDivisionError()))
    errors = self.flushLoggedErrors(ZeroDivisionError)
    self.assertEqual(1, len(errors))
    d = self.timer.stopService()
    self.assertIdentical(self.successResultOf(d), None)
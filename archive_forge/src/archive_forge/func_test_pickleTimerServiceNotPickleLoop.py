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
def test_pickleTimerServiceNotPickleLoop(self):
    """
        When pickling L{internet.TimerService}, it won't pickle
        L{internet.TimerService._loop}.
        """
    timer = TimerService(1, fakeTargetFunction)
    timer.startService()
    dumpedTimer = pickle.dumps(timer)
    timer.stopService()
    loadedTimer = pickle.loads(dumpedTimer)
    nothing = object()
    value = getattr(loadedTimer, '_loop', nothing)
    self.assertIdentical(nothing, value)
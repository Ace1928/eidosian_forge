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
def test_stopServiceWhileDisconnecting(self):
    """
        Calling L{ClientService.stopService} twice after it has
        connected (that is, stopping it while it is disconnecting)
        returns a L{Deferred} each time that fires when the
        disconnection has completed.
        """
    clock = Clock()
    cq, service = self.makeReconnector(fireImmediately=False, clock=clock)
    cq.connectQueue[0].callback(None)
    firstStopDeferred = service.stopService()
    secondStopDeferred = service.stopService()
    cq.constructedProtocols[0].connectionLost(Failure(IndentationError()))
    self.successResultOf(firstStopDeferred)
    self.successResultOf(secondStopDeferred)
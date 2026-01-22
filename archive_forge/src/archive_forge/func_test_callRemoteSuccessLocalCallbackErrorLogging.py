import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
def test_callRemoteSuccessLocalCallbackErrorLogging(self):
    """
        If the last callback on the L{Deferred} returned by C{callRemote} (added
        by application code calling C{callRemote}) fails, the failure is passed
        to the sender's C{unhandledError} method.
        """
    self.sender.expectError()
    callResult = self.dispatcher.callRemote(Hello, hello=b'world')
    callResult.addCallback(lambda result: 1 // 0)
    self.dispatcher.ampBoxReceived(amp.AmpBox({b'hello': b'yay', b'print': b'ignored', b'_answer': b'1'}))
    self._localCallbackErrorLoggingTest(callResult)
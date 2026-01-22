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
def test_callRemoteErrorLocalCallbackErrorLogging(self):
    """
        Like L{test_callRemoteSuccessLocalCallbackErrorLogging}, but for the
        case where the L{Deferred} returned by C{callRemote} fails.
        """
    self.sender.expectError()
    callResult = self.dispatcher.callRemote(Hello, hello=b'world')
    callResult.addErrback(lambda result: 1 // 0)
    self.dispatcher.ampBoxReceived(amp.AmpBox({b'_error': b'1', b'_error_code': b'bugs', b'_error_description': b'stuff'}))
    self._localCallbackErrorLoggingTest(callResult)
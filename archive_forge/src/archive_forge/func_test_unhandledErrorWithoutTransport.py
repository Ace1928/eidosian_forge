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
def test_unhandledErrorWithoutTransport(self):
    """
        L{amp.BinaryBoxProtocol.unhandledError} completes without error when
        there is no associated transport.
        """
    protocol = amp.BinaryBoxProtocol(self)
    protocol.makeConnection(StringTransport())
    protocol.connectionLost(Failure(Exception('Simulated')))
    protocol.unhandledError(Failure(RuntimeError('Fake error')))
    self.assertEqual(1, len(self.flushLoggedErrors(RuntimeError)))
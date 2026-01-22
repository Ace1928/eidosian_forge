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
def test_basicStructuredEmit(self):
    """
        Verify that a call similar to basicLiteralEmit's is handled properly with
        high-level quoting and passing to Python methods, and that argument
        names are correctly handled.
        """
    L = []

    class StructuredHello(amp.AMP):

        def h(self, *a, **k):
            L.append((a, k))
            return dict(hello=b'aaa')
        Hello.responder(h)
    c, s, p = connectedServerAndClient(ServerClass=StructuredHello)
    c.callRemote(Hello, hello=b'hello test', mixedCase=b'mixed case arg test', dash_arg=b'x', underscore_arg=b'y').addCallback(L.append)
    p.flush()
    self.assertEqual(len(L), 2)
    self.assertEqual(L[0], ((), dict(hello=b'hello test', mixedCase=b'mixed case arg test', dash_arg=b'x', underscore_arg=b'y', From=s.transport.getPeer(), Print=None, optional=None)))
    self.assertEqual(L[1], dict(Print=None, hello=b'aaa'))
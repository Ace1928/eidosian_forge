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
def test_basicLiteralEmit(self):
    """
        Verify that the command dictionaries for a callRemoteN look correct
        after being serialized and parsed.
        """
    c, s, p = connectedServerAndClient()
    L = []
    s.ampBoxReceived = L.append
    c.callRemote(Hello, hello=b'hello test', mixedCase=b'mixed case arg test', dash_arg=b'x', underscore_arg=b'y')
    p.flush()
    self.assertEqual(len(L), 1)
    for k, v in [(b'_command', Hello.commandName), (b'hello', b'hello test'), (b'mixedCase', b'mixed case arg test'), (b'dash-arg', b'x'), (b'underscore_arg', b'y')]:
        self.assertEqual(L[-1].pop(k), v)
    L[-1].pop(b'_ask')
    self.assertEqual(L[-1], {})
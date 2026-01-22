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
def test_valueTooLong(self):
    """
        Verify that attempting to send value longer than 64k will immediately
        raise an exception.
        """
    c, s, p = connectedServerAndClient()
    x = b'H' * (65535 + 1)
    tl = self.assertRaises(amp.TooLong, c.sendHello, x)
    p.flush()
    self.assertFalse(tl.isKey)
    self.assertTrue(tl.isLocal)
    self.assertEqual(tl.keyName, b'hello')
    self.failUnlessIdentical(tl.value, x)
    self.assertIn(str(len(x)), repr(tl))
    self.assertIn('value', repr(tl))
    self.assertIn('hello', repr(tl))
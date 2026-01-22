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
def test_nonFloat(self):
    """
        L{amp.Float.toString} raises L{ValueError} if passed an object which
        is not a L{float}.
        """
    argument = amp.Float()
    self.assertRaises(ValueError, argument.toString, '1.234')
    self.assertRaises(ValueError, argument.toString, b'1.234')
    self.assertRaises(ValueError, argument.toString, 1234)
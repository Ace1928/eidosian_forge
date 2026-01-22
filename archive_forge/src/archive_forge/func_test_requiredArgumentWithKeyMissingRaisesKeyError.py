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
def test_requiredArgumentWithKeyMissingRaisesKeyError(self):
    """
        L{ListOf.toBox} raises C{KeyError} if the argument's key is not
        present in the objects dictionary.
        """
    stringList = amp.ListOf(amp.Integer())
    self.assertRaises(KeyError, stringList.toBox, b'ommited', amp.AmpBox(), {'someOtherKey': 0}, None)
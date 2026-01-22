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
def test_responderDecorator(self):
    """
        A method on a L{CommandLocator} subclass decorated with a L{Command}
        subclass's L{responder} decorator should be returned from
        locateResponder, wrapped in logic to serialize and deserialize its
        arguments.
        """
    return self._checkSimpleGreeting(TestLocator, 8)
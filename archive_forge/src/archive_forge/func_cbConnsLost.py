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
def cbConnsLost(info):
    (serverSuccess, serverData), (clientSuccess, clientData) = info
    self.assertTrue(serverSuccess)
    self.assertTrue(clientSuccess)
    self.assertEqual(b''.join(serverData), SWITCH_CLIENT_DATA)
    self.assertEqual(b''.join(clientData), SWITCH_SERVER_DATA)
    self.testSucceeded = True
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
def switchit(self, name):
    if name == b'test-proto':
        self.maybeLaterProto = TestProto(self.onConnLost, SWITCH_SERVER_DATA)
        self.maybeLater = defer.Deferred()
        return self.maybeLater
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
def switchToTestProtocol(self, fail=False):
    if fail:
        name = b'no-proto'
    else:
        name = b'test-proto'
    p = TestProto(self.onConnLost, SWITCH_CLIENT_DATA)
    return self.callRemote(TestSwitchProto, SingleUseFactory(p), name=name).addCallback(lambda ign: p)
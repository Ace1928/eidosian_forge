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
def test_innerProtocolInRepr(self):
    """
        Verify that L{AMP} objects output their innerProtocol when set.
        """
    otherProto = TestProto(None, b'outgoing data')
    a = amp.AMP()
    a.innerProtocol = otherProto
    self.assertEqual(repr(a), '<AMP inner <TestProto #%d> at 0x%x>' % (otherProto.instanceId, id(a)))
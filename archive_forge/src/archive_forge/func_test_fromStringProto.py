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
def test_fromStringProto(self):
    """
        L{Descriptor.fromStringProto} constructs a file descriptor value by
        extracting a previously received file descriptor corresponding to the
        wire value of the argument from the L{_DescriptorExchanger} state of the
        protocol passed to it.

        This is a whitebox test which involves direct L{_DescriptorExchanger}
        state inspection.
        """
    argument = amp.Descriptor()
    self.protocol.fileDescriptorReceived(5)
    self.protocol.fileDescriptorReceived(3)
    self.protocol.fileDescriptorReceived(1)
    self.assertEqual(5, argument.fromStringProto('0', self.protocol))
    self.assertEqual(3, argument.fromStringProto('1', self.protocol))
    self.assertEqual(1, argument.fromStringProto('2', self.protocol))
    self.assertEqual({}, self.protocol._descriptors)
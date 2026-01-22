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
def test_makeArgumentsUsesCommandType(self):
    """
        L{amp.Command.makeArguments}'s return type should be the type
        of the result of L{amp.Command.commandType}.
        """
    protocol = object()
    objects = {'weird': b'whatever'}
    result = ProtocolIncludingCommandWithDifferentCommandType.makeArguments(objects, protocol)
    self.assertIs(type(result), MyBox)
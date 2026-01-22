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
def test_makeArguments(self):
    """
        There should be a class method of L{amp.Command} which accepts
        a mapping of argument names to objects and returns a similar
        mapping whose values have been serialized via the command's
        argument schema.
        """
    protocol = object()
    argument = object()
    objects = {'weird': argument}
    ident = '%d:%d' % (id(argument), id(protocol))
    self.assertEqual(ProtocolIncludingCommand.makeArguments(objects, protocol), {b'weird': ident.encode('ascii')})
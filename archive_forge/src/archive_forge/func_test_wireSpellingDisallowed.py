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
def test_wireSpellingDisallowed(self):
    """
        If a command argument conflicts with a Python keyword, the
        untransformed argument name is not allowed as a key in the dictionary
        passed to L{Command.makeArguments}.  If it is supplied,
        L{amp.InvalidSignature} is raised.

        This may be a pointless implementation restriction which may be lifted.
        The current behavior is tested to verify that such arguments are not
        silently dropped on the floor (the previous behavior).
        """
    self.assertRaises(amp.InvalidSignature, Hello.makeArguments, dict(hello='required', **{'print': 'print value'}), None)
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
def test_stringMessageReplacesNonAsciiText(self):
    """
        When C{errorCode} contains non-ASCII characters, L{amp.RemoteAmpError}
        renders then as backslash-escape sequences.
        """
    error = amp.RemoteAmpError(b'BROKEN-\xff', 'Something has broken')
    self.assertEqual('Code<BROKEN-\\xff>: Something has broken', str(error))
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
def test_stringMessage(self):
    """
        L{amp.RemoteAmpError} renders the given C{errorCode} (C{bytes}) and
        C{description} into a native string.
        """
    error = amp.RemoteAmpError(b'BROKEN', 'Something has broken')
    self.assertEqual('Code<BROKEN>: Something has broken', str(error))
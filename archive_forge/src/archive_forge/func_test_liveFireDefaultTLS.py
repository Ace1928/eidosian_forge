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
def test_liveFireDefaultTLS(self):
    """
        Verify that out of the box, we can start TLS to at least encrypt the
        connection, even if we don't have any certificates to use.
        """

    def secured(result):
        return self.cli.callRemote(SecuredPing)
    return self.cli.callRemote(amp.StartTLS).addCallback(secured)
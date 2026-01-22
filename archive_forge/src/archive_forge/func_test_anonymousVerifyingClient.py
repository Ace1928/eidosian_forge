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
def test_anonymousVerifyingClient(self):
    """
        Verify that anonymous clients can verify server certificates.
        """

    def secured(result):
        return self.cli.callRemote(SecuredPing)
    return self.cli.callRemote(amp.StartTLS, tls_verifyAuthorities=[tempcert]).addCallback(secured)
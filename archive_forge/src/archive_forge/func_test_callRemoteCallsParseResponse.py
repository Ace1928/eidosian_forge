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
def test_callRemoteCallsParseResponse(self):
    """
        Making a remote call on a L{amp.Command} subclass which
        overrides the C{parseResponse} method should call that
        C{parseResponse} method to get the response.
        """
    client = NoNetworkProtocol()
    thingy = b'weeoo'
    response = client.callRemote(MagicSchemaCommand, weird=thingy)

    def gotResponse(ign):
        self.assertEqual(client.parseResponseArguments, ({'weird': thingy}, client))
    response.addCallback(gotResponse)
    return response
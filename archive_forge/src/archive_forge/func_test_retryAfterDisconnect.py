import base64
import inspect
import re
from io import BytesIO
from typing import Any, List, Optional, Tuple, Type
from zope.interface import directlyProvides, implementer
import twisted.cred.checkers
import twisted.cred.credentials
import twisted.cred.error
import twisted.cred.portal
from twisted import cred
from twisted.cred.checkers import AllowAnonymousAccess, ICredentialsChecker
from twisted.cred.credentials import IAnonymous
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import address, defer, error, interfaces, protocol, reactor, task
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.mail import smtp
from twisted.mail._cred import LOGINCredentials
from twisted.protocols import basic, loopback
from twisted.python.util import LineLog
from twisted.trial.unittest import TestCase
def test_retryAfterDisconnect(self):
    """
        If the protocol created by L{SMTPSenderFactory} loses its connection
        before receiving confirmation of message delivery, it reconnects and
        tries to deliver the message again.
        """
    recipient = b'alice'
    message = b'some message text'
    domain = DummyDomain([recipient])

    class CleanSMTP(smtp.SMTP):
        """
            An SMTP subclass which ensures that its transport will be
            disconnected before the test ends.
            """

        def makeConnection(innerSelf, transport):
            self.addCleanup(transport.loseConnection)
            smtp.SMTP.makeConnection(innerSelf, transport)
    serverFactory = MultipleDeliveryFactorySMTPServerFactory([BrokenMessage, lambda user: DummyMessage(domain, user)])
    serverFactory.protocol = CleanSMTP
    serverPort = reactor.listenTCP(0, serverFactory, interface='127.0.0.1')
    serverHost = serverPort.getHost()
    self.addCleanup(serverPort.stopListening)
    sentDeferred = defer.Deferred()
    clientFactory = smtp.SMTPSenderFactory(b'bob@example.org', recipient + b'@example.com', BytesIO(message), sentDeferred)
    clientFactory.domain = b'example.org'
    clientConnector = reactor.connectTCP(serverHost.host, serverHost.port, clientFactory)
    self.addCleanup(clientConnector.disconnect)

    def cbSent(ignored):
        """
            Verify that the message was successfully delivered and flush the
            error which caused the first attempt to fail.
            """
        self.assertEqual(domain.messages, {recipient: [b'\n' + message + b'\n']})
        self.assertEqual(len(self.flushLoggedErrors(RuntimeError)), 1)
    sentDeferred.addCallback(cbSent)
    return sentDeferred
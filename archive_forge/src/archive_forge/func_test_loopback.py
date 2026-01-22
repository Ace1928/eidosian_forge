import base64
import hmac
import itertools
from collections import OrderedDict
from hashlib import md5
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
import twisted.cred.checkers
import twisted.cred.portal
import twisted.internet.protocol
import twisted.mail.pop3
import twisted.mail.protocols
from twisted import cred, internet, mail
from twisted.cred.credentials import IUsernameHashedPassword
from twisted.internet import defer
from twisted.internet.testing import LineSendingProtocol
from twisted.mail import pop3
from twisted.protocols import loopback
from twisted.python import failure
from twisted.trial import unittest, util
def test_loopback(self):
    """
        Messages can be downloaded over a loopback connection.
        """
    protocol = MyVirtualPOP3()
    protocol.service = self.factory
    clientProtocol = MyPOP3Downloader()

    def check(ignored):
        self.assertEqual(clientProtocol.message, self.message)
        protocol.connectionLost(failure.Failure(Exception('Test harness disconnect')))
    d = loopback.loopbackAsync(protocol, clientProtocol)
    return d.addCallback(check)
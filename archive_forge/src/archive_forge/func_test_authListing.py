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
def test_authListing(self):
    """
        L{pop3.POP3} responds to an I{AUTH} command with a list of supported
        authentication types based on its factory's C{challengers}.
        """
    p = DummyPOP3()
    p.factory = internet.protocol.Factory()
    p.factory.challengers = {b'Auth1': None, b'secondAuth': None, b'authLast': None}
    client = LineSendingProtocol([b'AUTH', b'QUIT'])
    d = loopback.loopbackAsync(p, client)
    return d.addCallback(self._cbTestAuthListing, client)
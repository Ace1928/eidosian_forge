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
def test_RetrieveUpdatesHighest(self):
    """
        Test that issuing a RETR command updates the LAST response.
        """
    p = self.pop3Server
    s = self.pop3Transport
    p.mbox.messages.append(self.extraMessage)
    p.lineReceived(b'RETR 2')
    self._flush()
    s.seek(0)
    s.truncate(0)
    p.lineReceived(b'LAST')
    self.assertEqual(s.getvalue(), b'+OK 2\r\n')
    s.seek(0)
    s.truncate(0)
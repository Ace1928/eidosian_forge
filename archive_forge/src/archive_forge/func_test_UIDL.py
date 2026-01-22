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
def test_UIDL(self):
    """
        Test the two forms of the UIDL command.  These are just like the two
        forms of the LIST command.
        """
    p = self.pop3Server
    s = self.pop3Transport
    p.lineReceived(b'UIDL 1')
    self.assertEqual(s.getvalue(), b'+OK 0\r\n')
    s.seek(0)
    s.truncate(0)
    p.lineReceived(b'UIDL')
    self._flush()
    self.assertEqual(s.getvalue(), b'+OK \r\n1 0\r\n.\r\n')
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
def run_PASS_before_USER(self, password):
    """
        Test protocol violation produced by calling PASS before USER.
        @type password: L{bytes}
        @param password: A password to test.
        """
    return self.runTest([b' '.join([b'PASS', password]), b'QUIT'], [b'+OK <moshez>', b'-ERR USER required before PASS', b'+OK '])
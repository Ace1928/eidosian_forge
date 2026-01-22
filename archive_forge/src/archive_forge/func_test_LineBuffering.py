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
def test_LineBuffering(self):
    """
        Test creating a LineBuffer and feeding it some lines.  The lines should
        build up in its internal buffer for a while and then get spat out to
        the writer.
        """
    output = []
    input = iter(itertools.cycle(['012', '345', '6', '7', '8', '9']))
    c = pop3._IteratorBuffer(output.extend, input, 6)
    i = iter(c)
    self.assertEqual(output, [])
    next(i)
    self.assertEqual(output, [])
    next(i)
    self.assertEqual(output, [])
    next(i)
    self.assertEqual(output, ['012', '345', '6'])
    for n in range(5):
        next(i)
    self.assertEqual(output, ['012', '345', '6', '7', '8', '9', '012', '345'])
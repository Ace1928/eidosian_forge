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
def contained(testcase, s, *caps):
    """
    Assert that the given capability is included in all of the capability
    sets.

    @param testcase: A L{unittest.TestCase} to use to make assertions.

    @param s: The capability for which to check.
    @type s: L{bytes}

    @param caps: The capability sets in which to check.
    @type caps: L{tuple} of iterable
    """
    for c in caps:
        testcase.assertIn(s, c)
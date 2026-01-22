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
def test_ParsingRoundTrip(self):
    """
        Verify that various kinds of data make it through the encode/parse
        round-trip unharmed.
        """
    c, s, p = connectedServerAndClient(ClientClass=LiteralAmp, ServerClass=LiteralAmp)
    SIMPLE = (b'simple', b'test')
    CE = (b'ceq', b': ')
    CR = (b'crtest', b'test\r')
    LF = (b'lftest', b'hello\n')
    NEWLINE = (b'newline', b'test\r\none\r\ntwo')
    NEWLINE2 = (b'newline2', b'test\r\none\r\n two')
    BODYTEST = (b'body', b'blah\r\n\r\ntesttest')
    testData = [[SIMPLE], [SIMPLE, BODYTEST], [SIMPLE, CE], [SIMPLE, CR], [SIMPLE, CE, CR, LF], [CE, CR, LF], [SIMPLE, NEWLINE, CE, NEWLINE2], [BODYTEST, SIMPLE, NEWLINE]]
    for test in testData:
        jb = amp.Box()
        jb.update(dict(test))
        jb._sendTo(c)
        p.flush()
        self.assertEqual(s.boxes[-1], jb)
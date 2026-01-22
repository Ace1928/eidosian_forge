import datetime
import itertools
import sys
from unittest import skipIf
from zope.interface import implementer
from incremental import Version
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet._idna import _idnaText
from twisted.internet.error import CertificateError, ConnectionClosed, ConnectionLost
from twisted.internet.task import Clock
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.reflect import requireModule
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_twisted import SetAsideModule
from twisted.trial import util
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
def test_NPNAndALPNSuccess(self):
    """
        When both ALPN and NPN are used, and both the client and server have
        overlapping protocol choices, a protocol is successfully negotiated.
        Further, the negotiated protocol is the first one in the list.
        """
    protocols = [b'h2', b'http/1.1']
    negotiatedProtocol, lostReason = negotiateProtocol(clientProtocols=protocols, serverProtocols=protocols)
    self.assertEqual(negotiatedProtocol, b'h2')
    self.assertIsNone(lostReason)
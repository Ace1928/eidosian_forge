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
@skipIf(skipSNI, skipSNI)
def test_hostnameIsIndicated(self):
    """
        Specifying the C{hostname} argument to L{CertificateOptions} also sets
        the U{Server Name Extension
        <https://en.wikipedia.org/wiki/Server_Name_Indication>} TLS indication
        field to the correct value.
        """
    names = []

    def setupServerContext(ctx):

        def servername_received(conn):
            names.append(conn.get_servername().decode('ascii'))
        ctx.set_tlsext_servername_callback(servername_received)
    cProto, sProto, cWrapped, sWrapped, pump = self.serviceIdentitySetup('valid.example.com', 'valid.example.com', setupServerContext)
    self.assertEqual(names, ['valid.example.com'])
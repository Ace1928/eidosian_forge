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
def test_certificateOptionsSerialization(self):
    """
        Test that __setstate__(__getstate__()) round-trips properly.
        """
    firstOpts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, method=SSL.SSLv23_METHOD, verify=True, caCerts=[self.sCert], verifyDepth=2, requireCertificate=False, verifyOnce=False, enableSingleUseKeys=False, enableSessions=False, fixBrokenPeers=True, enableSessionTickets=True)
    context = firstOpts.getContext()
    self.assertIs(context, firstOpts._context)
    self.assertIsNotNone(context)
    state = firstOpts.__getstate__()
    self.assertNotIn('_context', state)
    opts = sslverify.OpenSSLCertificateOptions()
    opts.__setstate__(state)
    self.assertEqual(opts.privateKey, self.sKey)
    self.assertEqual(opts.certificate, self.sCert)
    self.assertEqual(opts.method, SSL.SSLv23_METHOD)
    self.assertTrue(opts.verify)
    self.assertEqual(opts.caCerts, [self.sCert])
    self.assertEqual(opts.verifyDepth, 2)
    self.assertFalse(opts.requireCertificate)
    self.assertFalse(opts.verifyOnce)
    self.assertFalse(opts.enableSingleUseKeys)
    self.assertFalse(opts.enableSessions)
    self.assertTrue(opts.fixBrokenPeers)
    self.assertTrue(opts.enableSessionTickets)
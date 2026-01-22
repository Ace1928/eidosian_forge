import os
import hamcrest
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import waitUntilAllDisconnected
from twisted.protocols import basic
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test.test_tcp import ProperlyCloseFilesMixin
from twisted.trial.unittest import TestCase
from zope.interface import implementer
def test_missingCertificateFile(self):
    """
        Instantiating L{ssl.DefaultOpenSSLContextFactory} with a certificate
        filename which does not identify an existing file results in the
        initializer raising L{OpenSSL.SSL.Error}.
        """
    self.assertRaises(SSL.Error, ssl.DefaultOpenSSLContextFactory, certPath, self.mktemp())
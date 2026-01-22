from hashlib import sha1
from unittest import skipIf
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import client, error, jid, xmlstream
from twisted.words.protocols.jabber.sasl import SASLInitiatingInitializer
from twisted.words.xish import utility
@skipIf(*skipWhenNoSSL)
def test_tlsConfiguration(self):
    """
        A TLS configuration is passed to the TLS initializer.
        """
    configs = []

    def init(self, xs, required=True, configurationForTLS=None):
        configs.append(configurationForTLS)
    self.client_jid = jid.JID('user@example.com/resource')
    configurationForTLS = ssl.CertificateOptions()
    factory = client.XMPPClientFactory(self.client_jid, 'secret', configurationForTLS=configurationForTLS)
    self.patch(xmlstream.TLSInitiatingInitializer, '__init__', init)
    xs = factory.buildProtocol(None)
    version, tls, sasl, bind, session = xs.initializers
    self.assertIsInstance(tls, xmlstream.TLSInitiatingInitializer)
    self.assertIs(configurationForTLS, configs[0])
import random
from zope.interface.verify import verifyObject
from twisted.internet import defer, protocol
from twisted.internet.error import DNSLookupError, ServiceNameUnknownError
from twisted.internet.interfaces import IConnector
from twisted.internet.testing import MemoryReactor
from twisted.names import client, dns, srvconnect
from twisted.names.common import ResolverBase
from twisted.names.error import DNSNameError
from twisted.trial import unittest
def test_SRVLookupName(self):
    """
        The lookup name is a native string from service, protocol and domain.
        """
    client.theResolver.results = []
    self.connector.connect()
    name = client.theResolver.lookups[-1][0]
    self.assertEqual(b'_xmpp-server._tcp.example.org', name)
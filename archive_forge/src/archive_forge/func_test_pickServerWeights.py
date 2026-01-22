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
def test_pickServerWeights(self):
    """
        pickServer calculates running sum of weights and calls randint.

        This exercises the server selection algorithm specified in RFC 2782 by
        preparing fake L{random.randint} results and checking the values it was
        called with.
        """
    record1 = dns.Record_SRV(10, 10, 5222, 'host1.example.org')
    record2 = dns.Record_SRV(10, 20, 5222, 'host2.example.org')
    self.connector.orderedServers = [record1, record2]
    self.connector.servers = []
    self.patch(random, 'randint', self._randint)
    self.randIntResults = [11, 0]
    self.connector.pickServer()
    self.assertEqual(self.randIntArgs[0], (0, 30))
    self.connector.pickServer()
    self.assertEqual(self.randIntArgs[1], (0, 10))
    self.randIntResults = [10, 0]
    self.connector.pickServer()
    self.assertEqual(self.randIntArgs[2], (0, 30))
    self.connector.pickServer()
    self.assertEqual(self.randIntArgs[3], (0, 20))
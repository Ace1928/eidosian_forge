import copy
import operator
import socket
from functools import partial, reduce
from io import BytesIO
from struct import pack
from twisted.internet import defer, error, reactor
from twisted.internet.defer import succeed
from twisted.internet.testing import (
from twisted.names import authority, client, common, dns, server
from twisted.names.client import Resolver
from twisted.names.dns import SOA, Message, Query, Record_A, Record_SOA, RRHeader
from twisted.names.error import DomainError
from twisted.names.secondary import SecondaryAuthority, SecondaryAuthorityService
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.trial import unittest
def test_zoneTransfer(self):
    """
        Test DNS 'AXFR' queries (Zone transfer)
        """
    default_ttl = soa_record.expire
    results = [copy.copy(r) for r in reduce(operator.add, test_domain_com.records.values())]
    for r in results:
        if r.ttl is None:
            r.ttl = default_ttl
    return self.namesTest(self.resolver.lookupZone('test-domain.com').addCallback(lambda r: (r[0][:-1],)), results)
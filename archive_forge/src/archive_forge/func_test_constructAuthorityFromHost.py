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
def test_constructAuthorityFromHost(self):
    """
        L{SecondaryAuthorityService} can be constructed with a C{str} giving a
        master server address and several domains, causing the creation of a
        secondary authority for each domain and that master server address and
        the default DNS port.
        """
    primary = '192.168.1.2'
    service = SecondaryAuthorityService(primary, [b'example.com', 'example.org'])
    self.assertEqual(service.primary, primary)
    self.assertEqual(service._port, 53)
    self.assertEqual(service.domains[0].primary, primary)
    self.assertEqual(service.domains[0]._port, 53)
    self.assertEqual(service.domains[0].domain, b'example.com')
    self.assertEqual(service.domains[1].primary, primary)
    self.assertEqual(service.domains[1]._port, 53)
    self.assertEqual(service.domains[1].domain, b'example.org')
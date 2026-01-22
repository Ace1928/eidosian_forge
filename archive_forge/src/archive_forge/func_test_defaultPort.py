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
def test_defaultPort(self):
    """
        When constructed using L{SecondaryAuthority.__init__}, the default port
        of 53 is used.
        """
    secondary = SecondaryAuthority('192.168.1.1', 'inside.com')
    self.assertEqual(secondary.primary, '192.168.1.1')
    self.assertEqual(secondary._port, 53)
    self.assertEqual(secondary.domain, b'inside.com')
import binascii
import re
import string
import struct
import types
from hashlib import md5, sha1, sha256, sha384, sha512
from typing import Dict, List, Optional, Tuple, Type
from twisted import __version__ as twisted_version
from twisted.conch.error import ConchError
from twisted.conch.ssh import _kex, address, service
from twisted.internet import defer
from twisted.protocols import loopback
from twisted.python import randbytes
from twisted.python.compat import iterbytes
from twisted.python.randbytes import insecureRandom
from twisted.python.reflect import requireModule
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
def test_compatabilityVersion(self):
    """
        Test that the transport treats the compatibility version (1.99)
        as equivalent to version 2.0.
        """
    proto = MockTransportBase()
    proto.makeConnection(proto_helpers.StringTransport())
    proto.dataReceived(b'SSH-1.99-OpenSSH\n')
    self.assertTrue(proto.gotVersion)
    self.assertEqual(proto.otherVersionString, b'SSH-1.99-OpenSSH')
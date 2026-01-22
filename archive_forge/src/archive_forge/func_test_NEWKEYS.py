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
def test_NEWKEYS(self):
    """
        Test that NEWKEYS transitions the keys from nextEncryptions to
        currentEncryptions.
        """
    self.test_KEXINITMultipleAlgorithms()
    secure = [False]

    def stubConnectionSecure():
        secure[0] = True
    self.proto.connectionSecure = stubConnectionSecure
    self.proto.nextEncryptions = transport.SSHCiphers(b'none', b'none', b'none', b'none')
    self.simulateKeyExchange(b'AB', b'CD')
    self.assertIsNot(self.proto.currentEncryptions, self.proto.nextEncryptions)
    self.proto.nextEncryptions = MockCipher()
    self.proto.ssh_NEWKEYS(b'')
    self.assertIsNone(self.proto.outgoingCompression)
    self.assertIsNone(self.proto.incomingCompression)
    self.assertIs(self.proto.currentEncryptions, self.proto.nextEncryptions)
    self.assertTrue(secure[0])
    self.proto.outgoingCompressionType = b'zlib'
    self.simulateKeyExchange(b'AB', b'GH')
    self.proto.ssh_NEWKEYS(b'')
    self.assertIsNotNone(self.proto.outgoingCompression)
    self.proto.incomingCompressionType = b'zlib'
    self.simulateKeyExchange(b'AB', b'IJ')
    self.proto.ssh_NEWKEYS(b'')
    self.assertIsNotNone(self.proto.incomingCompression)
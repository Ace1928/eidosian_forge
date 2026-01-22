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
def test_keySetup(self):
    """
        Test that _keySetup sets up the next encryption keys.
        """
    self.proto.kexAlg = b'diffie-hellman-group14-sha1'
    self.proto.nextEncryptions = MockCipher()
    self.simulateKeyExchange(b'AB', b'CD')
    self.assertEqual(self.proto.sessionID, b'CD')
    self.simulateKeyExchange(b'AB', b'EF')
    self.assertEqual(self.proto.sessionID, b'CD')
    self.assertEqual(self.packets[-1], (transport.MSG_NEWKEYS, b''))
    newKeys = [self.proto._getKey(c, b'AB', b'EF') for c in iterbytes(b'ABCDEF')]
    self.assertEqual(self.proto.nextEncryptions.keys, (newKeys[0], newKeys[2], newKeys[1], newKeys[3], newKeys[4], newKeys[5]))
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
def test_keySetupWithExtInfo(self):
    """
        If the client advertised support for extension negotiation, then
        _keySetup sends SSH_MSG_EXT_INFO with the "server-sig-algs"
        extension as the next packet following the server's first
        SSH_MSG_NEWKEYS.  See RFC 8308, sections 2.4 and 3.1.
        """
    self.proto.supportedPublicKeys = [b'ssh-rsa', b'rsa-sha2-256', b'rsa-sha2-512']
    self.proto.kexAlg = b'diffie-hellman-group14-sha1'
    self.proto.nextEncryptions = MockCipher()
    self.proto._peerSupportsExtensions = True
    self.simulateKeyExchange(b'AB', b'CD')
    self.assertEqual(self.packets[-2], (transport.MSG_NEWKEYS, b''))
    self.assertEqual(self.packets[-1], (transport.MSG_EXT_INFO, b'\x00\x00\x00\x01' + common.NS(b'server-sig-algs') + common.NS(b'ssh-rsa,rsa-sha2-256,rsa-sha2-512')))
    self.simulateKeyExchange(b'AB', b'EF')
    self.assertEqual(self.packets[-1], (transport.MSG_NEWKEYS, b''))